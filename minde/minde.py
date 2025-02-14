import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from minde.libs.SDE import VP_SDE
from minde.libs.util import EMA,concat_vect, deconcat, marginalize_data, cond_x_data , get_samples, array_to_dataset, log_greyscale_images
from minde.libs.info_measures import mi_cond,mi_cond_sigma,mi_joint,mi_joint_sigma 
from minde.models.mlp import UnetMLP_simple
from minde.models.conv import UnetConv2D_simple
from minde.models.unet import UNet
from torch.utils.data import TensorDataset, DataLoader
from minde.scripts.helper import SynthetitcDataset

from mutinfo.estimators.base import MutualInformationEstimator

class MINDE(pl.LightningModule, MutualInformationEstimator):

    """
    Mutual Information Neural Diffusion Estimation.

    References
    ----------
    .. [1] G. Franzese, M. Bounoua and P. Michiardi, "MINDE: Mutual Information
        Neural Diffusion Estimation". ICLR, 2024.
    """
    
    def __init__(self, args):
                     
        super(MINDE, self).__init__()
        self.args = args
        if hasattr(self.args, 'gt'):
            self.gt = self.args.gt
        else:
            self.gt = None
        self.save_hyperparameters("args")
    
    def initialize_model(self):

        """
        Initialize the model.
        """

        if hasattr(self.args.model, 'hidden_dim')==False or self.args.model.hidden_dim == None:
            hidden_dim = self.calculate_hidden_dim()
        else:
            hidden_dim = self.args.model.hidden_dim
        
        if self.args.model.arch == "mlp":
            self.score = UnetMLP_simple(dim=np.sum(self.sizes), init_dim=hidden_dim, dim_mults=[],
                                        time_dim=hidden_dim, nb_var=2)
        elif self.args.model.arch == "conv": 
            self.score = UnetConv2D_simple(dim=sum(map(lambda x: x[0], self.sizes)), init_dim=hidden_dim, dim_mults=[], 
                                           time_dim=hidden_dim, nb_var=2)
        elif self.args.model.arch == "unet":
            assert self.sizes[0] == self.sizes[1], "The input variables must have the same size for the UNet architecture."
            self.score = UNet(sample_size=self.sizes[0], in_channels=2, out_channels=2, latent_dim=hidden_dim, norm_num_groups=self.args.model.norm_num_groups)
        else:
            raise NotImplementedError
        self.model_ema = EMA(self.score, decay=0.9999) if self.args.model.use_ema else None

        self.sde = VP_SDE(importance_sampling=self.args.inference.importance_sampling,
                          var_sizes=self.sizes,type =self.args.inference.type
                          )
        self.resume_training = True
        if hasattr(self.args, 'checkpoint_path'):
            state_dict = torch.load(self.args.checkpoint_path)
            # raise UserWarning(f"state dict keys: {state_dict.keys()}, state dict of the score model: {state_dict['state_dict'].keys()}")
            self.load_state_dict(state_dict['state_dict'])
    
    def get_var_size(self, x: np.ndarray) -> int:
        if len(x.shape) == 2:
            x_size = x.shape[1]
        elif len(x.shape) == 3:
            x_size = [1, x.shape[1], x.shape[2]]
        else:
            x_size = x.shape[1:]
        return x_size
    
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

        self.original_shapes = {
            "X": x.shape[1:],
            "Y": y.shape[1:]
        }

        if self.args.model.arch == "mlp":
            x = x.reshape(x.shape[0], -1)
            y = y.reshape(y.shape[0], -1)
            self.new_shape = (x.shape[1]+y.shape[1],)
        else:
            if len(x.shape) == 3:
                x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
            if len(y.shape) == 3:
                y = y.reshape(y.shape[0], 1, y.shape[1], y.shape[2])
            assert x.shape[2:] == y.shape[2:], "The input images must have the same size except for the channel dimension."
            self.new_shape = (x.shape[1]+y.shape[1], *x.shape[2:])
            self.original_shapes["X"] = x.shape[1:]
            self.original_shapes["Y"] = y.shape[1:]


        try:
            x_size = self.get_var_size(x)
            y_size = self.get_var_size(y)
        except:
            raise ValueError("Expected x and y to be numpy arrays, instead got {} and {}".format(type(x), type(y)))

        self.var_list = {"X": x_size, "Y": y_size}
        self.sizes = list(self.var_list.values())
        self.var_list = list(self.var_list.keys())

        self.initialize_model()

        self._check_arguments(x, y)

        data_set = array_to_dataset(x, y)

        train_loader = DataLoader(data_set, batch_size=self.args.training.bs, shuffle=True)

        # Something like that:
        if self.resume_training:
            print("Entering fit")
            self.fit(train_loader)
        else:
            if hasattr(self.args.inference, 'checkpoint_dir'):
                save_dir = self.args.inference.checkpoint_dir
            elif hasattr(self.args.training, 'checkpoint_dir'):
                save_dir = self.args.training.checkpoint_dir
            else:
                save_dir = "checkpoints"
            logger=pl.loggers.TensorBoardLogger(save_dir=save_dir)
            args_dict = vars(self.args)
            try:
                logger.log_hyperparams(args_dict)
            except:
                args_dict = args_dict['_content']
                logger.log_hyperparams(args_dict)
            
            self.to("cuda" if self.args.training.accelerator == "gpu" else "cpu")
            self.logger_valid = logger

            if self.args.inference.generate_samples:
                samples = self.generate_samples(bs=16)
                log_greyscale_images(self.logger, samples, self.global_step, "Generated Samples")

        data = {"X": data_set.x, "Y": data_set.y}
        #trained_model = self.trainer(self.model, x, y, **self.trainer_kwargs)

        #
        # Estimating...
        #

        mi, mi_sigma = self.compute_mi(data)

        print(f"Estimated MI: {np.mean(mi)} ± {np.std(mi)}")
        print(f"Estimated MI (σ): {np.mean(mi_sigma)} ± {np.std(mi_sigma)}")

        return {"mi": mi, "mi_sigma": mi_sigma}

    def generate_samples(self, bs: int=16):
        if self.args.inference.type=="c":
            mask = torch.tensor([1,-1]).to(self.device)
        elif self.args.inference.type=="j":
            mask = torch.tensor([1,1]).to(self.device)
        else:
            raise NotImplementedError
        samples = self.sde.generate_samples(self.score_inference, input_shape=self.new_shape, mask=mask, bs=bs)
        samples = deconcat(samples, self.var_list, self.sizes)
        # raise UserWarning(f"Samples shape: {samples['X'].shape},{samples['Y'].shape}, type: {type(samples)}")

        samples["X"] = samples["X"].reshape(-1, *self.original_shapes["X"])
        samples["Y"] = samples["Y"].reshape(-1, *self.original_shapes["Y"])
        return samples
    
    def fit(self,train_loader,test_loader=None):

        if test_loader is None:
            test_loader = DataLoader(train_loader.dataset, batch_size=train_loader.batch_size, shuffle=False, num_workers=train_loader.num_workers)
        
        self.test_samples = get_samples(test_loader,device="cuda"if self.args.training.accelerator == "gpu" else "cpu")
        args = self.args
        CHECKPOINT_DIR = args.training.checkpoint_dir

        logger=pl.loggers.TensorBoardLogger(save_dir=CHECKPOINT_DIR)
        
        trainer = pl.Trainer(logger=logger,
                         default_root_dir=CHECKPOINT_DIR,
                         accelerator=self.args.training.accelerator,
                         devices=self.args.training.devices,
                         max_steps=self.args.training.max_steps, # profiler="pytorch",
                         val_check_interval=self.args.training.val_check_interval,
                         check_val_every_n_epoch=None)  
        
        if self.args.log_example_images:
            samples = get_samples(train_loader, device="cuda" if self.args.training.accelerator == "gpu" else "cpu", N=1)
            x = samples["X"][:,0]
            x = x / 2 + 0.5 # [-1,+1] -> [0,1]
            x = (x * 255.0) # [0,1] -> [0,255]
            x = x.clamp(0, 255).to(torch.uint8) # Clamp to [0,255] and cast to uint8
            logger.experiment.add_image("Example x", x, 0)

            y = samples["Y"][:,0]
            y = y / 2 + 0.5 # [-1,+1] -> [0,1]
            y = (y * 255.0) # [0,1] -> [0,255]
            y = y.clamp(0, 255).to(torch.uint8) # Clamp to [0,255] and cast to uint8
            logger.experiment.add_image("Example y", y, 0)
        
        args_dict = vars(self.args)['_content']
        logger.log_hyperparams(args_dict)

        trainer.fit(model=self, train_dataloaders=train_loader,
                val_dataloaders=test_loader)
    
    
    def on_fit_start(self):
        self.sde.set_device(self.device) 
        
    def training_step(self, batch, batch_idx):
        self.train()
        loss = self.sde.train_step(batch, self.score_forward).mean()
        self.log("loss", loss)
        return {"loss": loss}
    
    def on_validation_epoch_start(self):
        self.logged_denoised = False
        self.logged_generated = False

    def validation_step(self, batch, batch_idx):
        self.eval()

        with torch.no_grad():
            if self.args.return_denoised and not self.logged_denoised:
                loss, x_denoised, x_noisy = self.sde.train_step(batch, self.score_forward, return_denoised=self.args.return_denoised)
                assert x_denoised.shape[1] == 2, f"Expected x_denoised to have shape (n_samples, 2, 16, 16) but got {x_denoised.shape}"
                x_denoised = deconcat(x_denoised, self.var_list, self.sizes)
                x_noisy = deconcat(x_noisy, self.var_list, self.sizes)
                log_greyscale_images(self.logger, x_denoised, self.global_step, "Denoised Images", 16)
                log_greyscale_images(self.logger, batch, self.global_step, "Original Images", 16)
                log_greyscale_images(self.logger, x_noisy, self.global_step, "Noisy Images", 16)
                loss = loss.mean()
                self.logged_denoised = True
            else:
                loss = self.sde.train_step(batch, self.score_forward).mean()
            if self.args.inference.generate_samples and not self.logged_generated:
                samples = self.generate_samples(bs=16)
                log_greyscale_images(self.logger, samples, self.global_step, "Generated Samples", 16)
                self.logged_generated = True
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

        if self.args.model.arch == "mlp" or self.args.model.arch == "conv":
          
            # MLP network requires the multitime vector
            #t = t.expand(mask.size()) * mask.clip(0, 1)
            t = t.expand(t.shape[0],mask.size(-1)) 
          
            marg = (- mask).clip(0, 1) ## max <0 
            cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
             
            t = t * (1- cond)  + 0.0 * cond
            t = t* (1-marg) + 1 * marg

            return self.score(x, t=t, std=std)
        
        elif self.args.model.arch == "unet":
            t = t.expand(t.shape[0],mask.size(-1)) 
          
            marg = (- mask).clip(0, 1) ## max <0 
            cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
             
            t = t * (1- cond)  + 0.0 * cond
            t = t* (1-marg) + 1 * marg

            return self.score(x, timestep=t)
        
        else:
            raise NotImplementedError


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

        score = self.model_ema.module if self.args.model.use_ema else self.score
        with torch.no_grad():
            score.eval()
            
            if self.args.model.arch == "mlp" or self.args.model.arch == "conv":
                t = t.expand(t.shape[0],mask.size(-1)) 
          
                marg = (- mask).clip(0, 1) ## max <0 
                cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
             
                t = t * (1- cond)  + 0.0 * cond
                t = t* (1-marg) + 1 * marg

                
                return score(x, t=t, std=std)
            elif self.args.model.arch == "unet":
                t = t.expand(t.shape[0],mask.size(-1)) 
            
                marg = (- mask).clip(0, 1) ## max <0 
                cond = 1 - (mask.clip(0, 1)) - marg  ##mask ==0
                
                t = t * (1- cond)  + 0.0 * cond
                t = t* (1-marg) + 1 * marg

                return score(x, timestep=t)
            else:
                raise NotImplementedError

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.score.parameters(), lr=self.args.training.lr)
        return optimizer
    
    def on_train_batch_end(self, *args, **kwargs) -> None:
        if self.global_step % self.args.training.val_check_interval == 0:
            self.logger_estimates()

    def infer_scores(self,z_t,t, data_0, std_w,marg_masks,cond_mask):
        
        with torch.no_grad():
            if self.args.inference.type=="c":
                
                marg_x = concat_vect(marginalize_data(z_t, self.var_list[0],fill_zeros=True))
                cond_x = concat_vect(cond_x_data(z_t, data_0, self.var_list[0]))
                
                s_marg = - self.score_inference(marg_x, t=t, mask=marg_masks[self.var_list[0]], std=std_w).detach()
                s_cond = - self.score_inference(cond_x, t=t, mask=cond_mask[self.var_list[0]], std=std_w).detach()
                return deconcat(s_marg,self.var_list,self.sizes)[self.var_list[0]] , deconcat(s_cond,self.var_list,self.sizes)[self.var_list[0]]
                
            elif self.args.inference.type=="j":
                
                s_joint = - self.score_inference( concat_vect(z_t), t=t, std=std_w, mask=torch.ones_like(marg_masks[self.var_list[0]])).detach()
                
                cond_x = concat_vect(cond_x_data(z_t, data_0, self.var_list[0]))
                cond_y = concat_vect(cond_x_data(z_t, data_0, self.var_list[1]))
                
                s_cond_x = - self.score_inference( cond_x, t=t, mask=cond_mask[self.var_list[0]], std=std_w).detach() ##S(X|Y)
                s_cond_y = - self.score_inference( cond_y, t=t, mask=cond_mask[self.var_list[1]], std=std_w).detach() ##S(Y|X)
                
                return s_joint,deconcat(s_cond_x,self.var_list,self.sizes)[self.var_list[0]], deconcat(s_cond_y,self.var_list,self.sizes)[self.var_list[1]]
            



    def compute_mi(self, data=None, eps=1e-5):
        """
        Compute mutual information.

        Args:
            data (dict): A dictionary containing the input data.{x0:  , x1: , x2: , ...}
           

        Returns:
            tuple: A tuple containing the computed mutual information (Difference inside and difference outside).

        """
        self.eval()
        self.to("cuda" if self.args.training.accelerator == "gpu" else "cpu")
        if data==None:
            data = self.test_samples
        self.sde.device = self.device
        var_list = list(data.keys())
        data_0 = {x_i: data[x_i].to(self.device) for x_i in var_list}
        z_0 = concat_vect(data_0)

        mi = []
        mi_sigma = []

        dataset = SynthetitcDataset(data_0)

        dataloader = DataLoader(dataset, batch_size=self.args.inference.bs, shuffle=False)

        for batch in dataloader:
            z_0 = concat_vect(batch)
            M = z_0.shape[0]
        
            marg_masks, cond_mask = self.get_masks(var_list)

            for i in range(self.args.inference.mc_iter):
                # Sample t
                if self.args.inference.importance_sampling:
                    t = (self.sde.sample_importance_sampling_t(
                        shape=(M, 1))).to(self.device)
                else:
                    t = ((self.sde.T - eps) * torch.rand((M, 1)) + eps).to(self.device)
                _, g = self.sde.sde(t)
                # Sample from the SDE (pertrbe the data with noise at time)
                z_t, _, mean, std = self.sde.sample(z_0, t=t)
                
                std_w = None if self.args.inference.importance_sampling else std 
                z_t = deconcat(z_t, self.var_list, self.sizes)

                
                if self.args.inference.type =="c":
                
                    s_marg, s_cond = self.infer_scores(z_t,t, batch, std_w, marg_masks, cond_mask)
                    mi.append(
                        mi_cond(s_marg=s_marg,s_cond=s_cond,g=g,importance_sampling=self.args.inference.importance_sampling)
                    )
                    mi_sigma.append(
                        mi_cond_sigma(s_marg=s_marg,s_cond=s_cond,
                                    g=g,mean=mean,std=std,x_t= z_t[self.var_list[0]],sigma=self.args.inference.sigma,
                                    importance_sampling=self.args.inference.importance_sampling)
                    )
                    
                elif self.args.inference.type=="j":
                    s_joint, s_cond_x,s_cond_y = self.infer_scores(z_t,t, batch, std_w, marg_masks, cond_mask)
                    mi.append(
                        mi_joint(s_joint=s_joint,
                                        s_cond_x=s_cond_x,
                                        s_cond_y=s_cond_y,g=g,importance_sampling=self.args.inference.importance_sampling)
                    )
                    mi_sigma.append(
                        mi_joint_sigma(s_joint=s_joint,
                                        s_cond_x=s_cond_x,
                                        s_cond_y=s_cond_y,
                                        x_t=z_t[self.var_list[0]],
                                        y_t=z_t[self.var_list[1]] ,
                                        g=g,mean=mean,std=std,
                                        sigma=self.args.inference.sigma,
                                        importance_sampling=self.args.inference.importance_sampling)
                    )
        print(f"Std is {np.std(mi)}, std sigma is {np.std(mi_sigma)}")
        return np.mean(mi), np.mean(mi_sigma)
    
    def compute_entropy(self, data=None, eps=1e-5):
        raise NotImplementedError("Entropy estimation is not implemented yet.")

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
        if self.args.model.arch == "mlp" or self.args.model.arch == "conv" or self.args.model.arch == "unet":
            dim = np.sum(self.sizes)
            if dim <= 10:
                hidden_dim = 64
            elif dim <= 50:
                hidden_dim = 128
            else:
                hidden_dim = 256
            return hidden_dim
        
    def logger_estimates(self):
        with torch.no_grad():
            self.eval()
            self.score.eval()
            mi, mi_sigma = self.compute_mi(data=self.test_samples)
            print("Step: ",self.global_step," GT: ",np.round( self.gt, decimals=3 )  if self.gt != None else "Not given", "MINDE_estimate: ",np.round( mi, decimals=3 ),"MINDE_sigma_estimate: ",np.round( mi_sigma, decimals=3 ) )

            self.logger.experiment.add_scalars('Measures/mi',
                                                   {'gt': self.gt if self.gt != None else 0,
                                                    'minde': np.mean(mi),"minde_sigma":np.mean(mi_sigma),
                                                    }, global_step=self.global_step)
            self.train()
            self.score.train()

    