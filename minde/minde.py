import numpy as np
import torch
import pytorch_lightning as pl
from minde.libs.SDE import VP_SDE
from minde.libs.util import EMA,concat_vect, deconcat, marginalize_data, cond_x_data , get_samples, array_to_dataset
from minde.libs.info_measures import mi_cond,mi_cond_sigma,mi_joint,mi_joint_sigma 
from minde.models.mlp import UnetMLP_simple
from torch.utils.data import DataLoader

from mutinfo.estimators.base import MutualInformationEstimator

class MINDE(pl.LightningModule, MutualInformationEstimator):

    """
    Mutual Information Neural Diffusion Estimation.

    References
    ----------
    .. [1] G. Franzese, M. Bounoua and P. Michiardi, "MINDE: Mutual Information
        Neural Diffusion Estimation". ICLR, 2024.
    """
    
    def __init__(self,args,gt=None,var_list = None):
                     
        super(MINDE, self).__init__()
        
        if var_list ==None:
            if not hasattr(args, 'dim'):
                raise ValueError("If var_list is not given, the dimension of the variables must be given.")
            var_list = {"x" + str(i): args.dim for i in range(2)}

        self.args = args
        self.var_list = list(var_list.keys())
        self.sizes = list(var_list.values())
        self.gt = gt
        
        
        self.save_hyperparameters("args")
        self.initialize_model()
    
    def initialize_model(self):

        """
        Initialize the model.
        """

        if hasattr(self.args, 'hidden_dim')==False or self.args.hidden_dim == None:
            hidden_dim = self.calculate_hidden_dim()
        else:
            hidden_dim = self.args.hidden_dim
        
        if self.args.arch == "mlp":
            self.score = UnetMLP_simple(dim=np.sum(self.sizes), init_dim=hidden_dim, dim_mults=[],
                                        time_dim=hidden_dim, nb_var=2)
        else:
            raise NotImplementedError
        self.model_ema = EMA(self.score, decay=0.999) if self.args.use_ema else None

        self.sde = VP_SDE(importance_sampling=self.args.importance_sampling,
                          var_sizes=self.sizes,type =self.args.type
                          )
    
    def __call__(self, x: np.ndarray, y: np.ndarray, std: bool=False, sigma: bool=False, all: bool=True, eps: float=1e-5) -> float:
        """
        Estimate the value of mutual information between two random vectors
        using samples `x` and `y`.

        Parameters
        ----------
        x : array_like
            Samples from the first random vector.
        y : array_like
            Samples from the second random vector.
        std : bool
            Calculate standard deviation.
        sigma : bool
            Return the estimate using the σ parameter.
        all : bool
            Return all the estimates as a dictionary.

        Returns
        -------
        mutual_information : float
            Estimated value of mutual information.
        mutual_information_std : float or None
            Standard deviation of the estimate, or None if `std=False`
        """

        self.initialize_model()

        self._check_arguments(x, y)

        data_set = array_to_dataset(x, y)
        train_loader = DataLoader(data_set, batch_size=self.args.bs, shuffle=True)

        # Something like that:
        self.fit(train_loader)

        data = {"x": data_set.x, "y": data_set.y}
        #trained_model = self.trainer(self.model, x, y, **self.trainer_kwargs)

        #
        # Estimating...
        #

        mean, std_, mean_sigma, std_sigma = self.compute_mi(data, return_std=True)

        print(f"Estimated MI: {mean} ± {std_}")
        print(f"Estimated MI (σ): {mean_sigma} ± {std_sigma}")

        if all:
            return {"mi": mean, "mi_std": std_, "mi_sigma": mean_sigma, "mi_sigma_std": std_sigma}

        if sigma:
            mean = mean_sigma
            std_ = std_sigma
        if std:
            return mean, std_
        else:
            return mean
    
    def fit(self,train_loader,test_loader=None):

        if test_loader is None:
            test_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers)
        
        self.test_samples = get_samples(test_loader,device="cuda"if self.args.accelerator == "gpu" else "cpu")
        args = self.args
        CHECKPOINT_DIR = args.checkpoint_dir
        
        trainer = pl.Trainer(logger=pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR),
                         default_root_dir=CHECKPOINT_DIR,
                         accelerator=self.args.accelerator,
                         devices=self.args.devices,
                         max_epochs=self.args.max_epochs, # profiler="pytorch",
                         check_val_every_n_epoch=self.args.check_val_every_n_epoch)  
    
        trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader)
    
    
    def on_fit_start(self):
        self.sde.set_device(self.device) 
        
    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.sde.train_step(batch, self.score_forward).mean()
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self.sde.train_step(batch, self.score_forward).mean()
        self.log("loss_test", loss)
        return {"loss": loss}

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.score)

    
    
    
    def score_forward(self, x, t=None, mask=None, std=None):
        """
        Perform score inference on the input data.

        Args:
            x (torch.Tensor): Concatenated variables.
            t (torch.Tensor, optional): The time t. 
            mask (torch.Tensor, optional): The mask data.
            std (torch.Tensor, optional): The standard deviation to rescale the network output.

        Returns:
            torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
        """

        if self.args.arch == "mlp":
          
            # MLP network requires the multitime vector
            #t = t.expand(mask.size()) * mask.clip(0, 1)
            t = t.expand(t.shape[0],mask.size(-1)) 
          
            marg = (- mask).clip(0, 1) ## max <0 
            cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
             
            t = t * (1- cond)  + 0.0 * cond
            t = t* (1-marg) + 1 * marg

            return self.score(x, t=t, std=std)


    def score_inference(self, x, t=None, mask=None, std=None):
        """
        Perform score inference on the input data.

        Args:
            x (torch.Tensor): Concatenated variables.
            t (torch.Tensor, optional): The time t. 
            mask (torch.Tensor, optional): The mask data.
            std (torch.Tensor, optional): The standard deviation to rescale the network output.

        Returns:
            torch.Tensor: The output score function (noise/std) if std !=None , else return noise .
        """
        # Get the model to use for inference, use the ema model if use_ema is set to True

        score = self.model_ema.module if self.args.use_ema else self.score
        with torch.no_grad():
            score.eval()
            
            if self.args.arch == "mlp":
                t = t.expand(t.shape[0],mask.size(-1)) 
          
                marg = (- mask).clip(0, 1) ## max <0 
                cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
             
                t = t * (1- cond)  + 0.0 * cond
                t = t* (1-marg) + 1 * marg

                
                return score(x, t=t, std=std)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.score.parameters(), lr=self.args.lr)
        return optimizer

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        if self.current_epoch % self.args.test_epoch == 0 and self.current_epoch != 0 and self.current_epoch> self.args.warmup_epochs:
            self.logger_estimates()

    def infer_scores(self,z_t,t, data_0, std_w,marg_masks,cond_mask):
        

        with torch.no_grad():
            if self.args.type=="c":
                
                marg_x = concat_vect(marginalize_data(z_t, self.var_list[0],fill_zeros=True))
                cond_x = concat_vect(cond_x_data(z_t, data_0, self.var_list[0]))
                
                s_marg = - self.score_inference(marg_x, t=t, mask=marg_masks[self.var_list[0]], std=std_w).detach()
                s_cond = - self.score_inference(cond_x, t=t, mask=cond_mask[self.var_list[0]], std=std_w).detach()
                return deconcat(s_marg,self.var_list,self.sizes)[self.var_list[0]] , deconcat(s_cond,self.var_list,self.sizes)[self.var_list[0]]
                
            elif self.args.type=="j":
                
                s_joint = - self.score_inference( concat_vect(z_t), t=t, std=std_w, mask=torch.ones_like(marg_masks[self.var_list[0]])).detach()
                
                cond_x = concat_vect(cond_x_data(z_t, data_0, self.var_list[0]))
                cond_y = concat_vect(cond_x_data(z_t, data_0, self.var_list[1]))
                
                s_cond_x = - self.score_inference( cond_x, t=t, mask=cond_mask[self.var_list[0]], std=std_w).detach() ##S(X|Y)
                s_cond_y = - self.score_inference( cond_y, t=t, mask=cond_mask[self.var_list[1]], std=std_w).detach() ##S(Y|X)
                
                return s_joint,deconcat(s_cond_x,self.var_list,self.sizes)[self.var_list[0]], deconcat(s_cond_y,self.var_list,self.sizes)[self.var_list[1]]
            



    def compute_mi(self, data=None, eps=1e-5, return_std=False):
        """
        Compute mutual information.

        Args:
            data (dict): A dictionary containing the input data.{x0:  , x1: , x2: , ...}
           

        Returns:
            tuple: A tuple containing the computed mutual information (Difference inside and difference outside).

        """
        self.eval()
        self.to("cuda" if self.args.accelerator == "gpu" else "cpu")
        if data==None:
            data = self.test_samples
        self.sde.device = self.device
        var_list = list(data.keys())
        data_0 = {x_i: data[x_i].to(self.device) for x_i in var_list}
        z_0 = concat_vect(data_0)

        N = len(self.sizes)
        M = z_0.shape[0]

        mi = []
        mi_sigma = []
        
        marg_masks, cond_mask = self.get_masks(var_list)

        for i in range(self.args.mc_iter):
            # Sample t
            if self.args.importance_sampling:
                t = (self.sde.sample_importance_sampling_t(
                    shape=(M, 1))).to(self.device)
            else:
                t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
            _, g = self.sde.sde(t)
            # Sample from the SDE (pertrbe the data with noise at time)
            z_t, _, mean, std = self.sde.sample(z_0, t=t)
            
            std_w = None if self.args.importance_sampling else std 
            z_t = deconcat(z_t, self.var_list, self.sizes)
            
            if self.args.type =="c":
              
                s_marg, s_cond = self.infer_scores(z_t,t, data_0, std_w, marg_masks, cond_mask)
                
                mi.append(
                    mi_cond(s_marg=s_marg,s_cond=s_cond,g=g,importance_sampling=self.args.importance_sampling)
                )
                mi_sigma.append(
                     mi_cond_sigma(s_marg=s_marg,s_cond=s_cond,
                                   g=g,mean=mean,std=std,x_t= z_t[self.var_list[0]],sigma=self.args.sigma,
                                   importance_sampling=self.args.importance_sampling)
                )
                
            elif self.args.type=="j":
                s_joint, s_cond_x,s_cond_y = self.infer_scores(z_t,t, data_0, std_w, marg_masks, cond_mask)
                mi.append(
                    mi_joint(s_joint=s_joint,
                                    s_cond_x=s_cond_x,
                                    s_cond_y=s_cond_y,g=g,importance_sampling=self.args.importance_sampling)
                )
                mi_sigma.append(
                     mi_joint_sigma(s_joint=s_joint,
                                    s_cond_x=s_cond_x,
                                    s_cond_y=s_cond_y,
                                    x_t= z_t[self.var_list[0]],
                                    y_t=z_t[self.var_list[1]] ,
                                    g=g,mean=mean,std=std,
                                    sigma=self.args.sigma,
                                    importance_sampling=self.args.importance_sampling)
                )
        mi = np.array(mi)
        mi_sigma = np.array(mi_sigma)
        # raise UserWarning(f"Shape of mi: {mi.shape}, shape of mi_sigma: {mi_sigma.shape}")
        avg_mi = np.mean(mi)
        avg_mi_sigma = np.mean(mi_sigma)
        std_mi = np.std(mi)
        std_mi_sigma = np.std(mi_sigma)

        if return_std:
            return avg_mi, std_mi, avg_mi_sigma, std_mi_sigma
        return avg_mi, avg_mi_sigma
    

    def get_masks(self, var_list):
        """_summary_
        Returns:
            dict , dict :  marginal masks, conditional masks 
        """
        return {var_list[0]: torch.tensor([1,-1]).to(self.device),
                var_list[1]: torch.tensor([-1,1]).to(self.device),
                },{var_list[0]: torch.tensor([1,0]).to(self.device),
                var_list[1]: torch.tensor([0,1]).to(self.device),
                }


    def calculate_hidden_dim(self):
        # return dimensions for the hidden layers
        if self.args.arch == "mlp":
            dim = np.sum(self.sizes)
            if dim <= 10:
                hidden_dim = 64
            elif dim <= 50:
                hidden_dim = 128
            else:
                hidden_dim = 256
            return hidden_dim
        



    def logger_estimates(self):
 
        mi, mi_sigma = self.compute_mi(data=self.test_samples)

        print("Epoch: ",self.current_epoch," GT: ",np.round( self.gt, decimals=3 )  if self.gt != None else "Not given", "MINDE_estimate: ",np.round( mi, decimals=3 ),"MINDE_sigma_estimate: ",np.round( mi_sigma, decimals=3 ) )

        self.logger.experiment.add_scalars('Measures/mi',
                                                   {'gt': self.gt if self.gt != None else 0,
                                                    'minde': mi,"minde_sigma":mi_sigma,
                                                    }, global_step=self.global_step)

    