
import pytorch_lightning as pl
import os
import jax
import json
from .helper import *
from minde.minde import MINDE
from minde.scripts.config import get_config
jax.config.update('jax_platform_name', 'cpu')

from mutinfo.distributions.base import UniformlyQuantized
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm  # Used as a base distribution (to be quantized), you can use any other having a `cdf` method.
from scipy.stats import bernoulli, poisson, binom, uniform, expon, t

parser = get_config()

def evaluate_task(args, sampler, transformation=None):

    if sampler == "bernoulli":
        X = bernoulli.rvs(p=0.5,size=args.n_samples)
        Y = np.copy(X)
    else:
        X, Y = sampler.rvs(args.n_samples)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

    if transformation is not None:
        X = transformation(X)
        Y = transformation(Y)

    if X.shape[0] <= 5:
        args.max_epochs = 300
        args.lr = 1e-3
        args.bs = 128
    elif X.shape[0] > 5:
        args.max_epochs = 500
        args.lr = 2e-3
        args.bs = 256
    

    var_list = {"x": X.shape[1], "y": Y.shape[1]}

    model = MINDE(args, var_list=var_list).to("cuda")

    print("Model created")
    
    _ = model(X, Y, std=True) # fit the model

    print("Model fitted")

    model.eval()
    results = []

    random_states = np.random.randint(0, 10000, args.test_runs)

    for i in range(args.test_runs):

        # Flatten the joint distribution

        if sampler == "bernoulli":
            X = bernoulli.rvs(p=0.5,size=args.n_samples_test, random_state = random_states[i])
            Y = np.copy(X)
        else:
            X, Y = sampler.rvs(args.n_samples_test, random_state = random_states[i])

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        if transformation is not None:
            X = transformation(X)
            Y = transformation(Y)

        X = StandardScaler(copy=True).fit_transform(X)
        Y = StandardScaler(copy=True).fit_transform(Y)

        data = {"x": torch.tensor(X, dtype=torch.float16).to(model.device), "y": torch.tensor(Y, dtype=torch.float16).to(model.device)}

        results.append(model.compute_mi(data))
    res_mi = [r[0] for r in results]
    res_mi_sigma = [r[1] for r in results]

    return {"mi": np.mean(res_mi), "mi_sigma": np.mean(res_mi_sigma)}




if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    return_dict = lambda x: {
        # "uniform": UniformlyQuantized(x, uniform(0, 1)),
        # "expon": UniformlyQuantized(x, expon(1)),
        # "norm": UniformlyQuantized(x, X_dim=args.dim_x, Y_dim=args.dim_y, base_rv=norm(0, 1)),
        "bernoulli": "bernoulli",
        # "t-student 1dof": UniformlyQuantized(x, t(1)),
        # "t-student 2dof": UniformlyQuantized(x, t(2)),
    }
    
    results_tasks = {}

    pl.seed_everything(args.seed)

    mutinfos = np.arange(0, 5.1, 1, dtype=float)
    mutinfos = [np.log(2)]
    # mutinfos = np.arange(0, 10.1, 0.5, dtype=float)

    transformation = lambda x: np.arcsinh(x)
    transformation = None

    for info in mutinfos:
        
        samplers_dict = return_dict(info)

        for name, sampler in samplers_dict.items():

            print(f"Sampler {name} with dimensions {args.dim_x} and {args.dim_y} mi: {info} ")

            results = evaluate_task(args, sampler, transformation)

            exp_name = f"{name}_{args.dim_x}_{args.dim_y}_{info}"

            results_tasks[exp_name] = {
                "gt": info,
                "minde_estimate": results,
                "minde type": args.type,
                "importance_sampling": args.importance_sampling,
                "arch": args.arch
            }
            print(results_tasks)

    directory = args.results_dir
    if not os.path.exists(directory):
        os.makedirs(directory)
    directory_path = directory+'/'.format(
        args.importance_sampling, args.arch, args.seed)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open("{}/results_bernoulli.json".format(directory_path), 'w') as f:
        json.dump(results_tasks, f)
