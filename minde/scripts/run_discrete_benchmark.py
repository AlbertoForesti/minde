
import pytorch_lightning as pl
import os
import jax
import json
from .helper import *
from minde.minde import MINDE
from minde.scripts.config import get_config
jax.config.update('jax_platform_name', 'cpu')

from mutinfo.distributions.base import UniformlyQuantized
from scipy.stats import norm  # Used as a base distribution (to be quantized), you can use any other having a `cdf` method.
from scipy.stats import bernoulli, poisson, binom, uniform, expon

parser = get_config()

def evaluate_task(args, sampler):

    X, Y = sampler.rvs(args.n_samples)

    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)

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

        X, Y = sampler.rvs(args.n_samples_test, random_state = random_states[i])

        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        data = {"x": torch.tensor(X, dtype=torch.float16).to(model.device), "y": torch.tensor(Y, dtype=torch.float16).to(model.device)}

        results.append(model.compute_mi(data))
    res_mi = [r[0] for r in results]
    res_mi_sigma = [r[1] for r in results]

    return {"mi": np.mean(res_mi), "mi_sigma": np.mean(res_mi_sigma)}




if __name__ == "__main__":
    args = parser.parse_args()    

    samplers_dict = {
        "bernoulli": UniformlyQuantized(1.0, bernoulli(0.5)),
        "poisson": UniformlyQuantized(1.0, poisson(10, 0.5)),
        "binomial": UniformlyQuantized(1.0, binom(10, 0.5)),
        "uniform": UniformlyQuantized(1.0, uniform(0, 1)),
        "expon": UniformlyQuantized(1.0, expon(1))
    }
    
    results_tasks = {}

    pl.seed_everything(args.seed)

    for name, sampler in samplers_dict.items():

        print(f"Sampler {name} with dimensions {args.dim_x} and {args.dim_y} mi: {1.0} ")

        results = evaluate_task(args, sampler)

        results_tasks[name] = {
            "gt": 1.0,
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

    with open("{}/results_discrete.json".format(directory_path), 'w') as f:
        json.dump(results_tasks, f)
