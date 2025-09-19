'''
Input: Neg Pos features from extract.py or extract_multi.py
Output: saved optimized cvxNN model on features extracted

This file serves to pass in hyperparams for CRONOS and adamW, easy for grid search plots later
defrun.py handles actual work, dataloading and input directories handled in dataloader utils
'''

'''
cronos_trainer.py needs to accept a model_name argument for use with run_cvxdpo_pipeline_simple.sh

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
args = parser.parse_args()
model_names = args.model_name

This file runs adamW with 3 seeds for plotting purposes
save statistics into a csv file or pandas df for line plot later
'''

import jax
import numpy as np
import jax.numpy as jnp
from solve.optimizers.admm import admm
import os
from os.path import dirname, join, abspath
import pickle
from defrun import run, RunResults  # returns a NamedTuple
import random
import pandas as pd
import wandb  # Added wandb import
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--target_lang', type=str, required=False, default='en')
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()
model_names = args.model_name

# List of model names; these are folders where POS and NEG features are saved per model
# the input data directory is mapped in gpt2_dataloader.py

# Tune rho, admm_iters, pcg_iters, take out jit for adamW
# pcg_iters most important
cronos_params = dict(
    rank=20, beta=0.001, rho=0.0001,
    gamma_ratio=1, admm_iters=6, pcg_iters=32,
    check_opt=False
)

adamW_params = dict(optimizer='AdamW', gamma=10**-4, n_epoch=30, batch_size=1024)

opt_seed = 1024
data_seed = random.randint(1, 10)

# Initialize wandb with all configuration at once
wandb.init(
    project="CLD",
    name=f"cronos_{model_names}",
    config={
        "model_name": model_names,
        "cronos_params": cronos_params,
        "adamW_params": adamW_params,
        "opt_seed": opt_seed,
        "data_seed": data_seed,
        "output_dir": args.output_dir,
        "rank": cronos_params["rank"],
        "beta": cronos_params["beta"],
        "rho": cronos_params["rho"],
        "gamma_ratio": cronos_params["gamma_ratio"],
        "admm_iters": cronos_params["admm_iters"],
        "pcg_iters": cronos_params["pcg_iters"],
        "optimizer": adamW_params["optimizer"],
        "learning_rate": adamW_params["gamma"],
        "n_epoch": adamW_params["n_epoch"],
        "batch_size": adamW_params["batch_size"]
    }
)

# Record start time for TFLOPS calculation
start_time = time.time()

# run model training and evaluation (returns NamedTuple)
# results is now a variable, and RunResults is the type
results: RunResults = run(model_names, args.data_dir, cronos_params, adamW_params, opt_seed, data_seed, args.output_dir, args.target_lang)

elapsed_time = time.time() - start_time

# This is a rough estimate based on RTX 4090 specs 
def estimate_tflops(duration_seconds):
    """
    Estimate TFLOPs for an NVIDIA RTX 4090 @ 70% bf16 Tensor Core efficiency
    """
    gflops_per_sec = 231000  # 231 TFLOPs = 70% of 330 peak bf16 performance
    tflops_used = (gflops_per_sec * duration_seconds) / 1000
    return tflops_used

estimated_tflops = estimate_tflops(elapsed_time)


wandb.log({
    "global_max_test_peak": results.global_max_test_peak,
    "global_delta_test_peak": results.global_delta_test_peak,
    "training_time": elapsed_time,
    "estimated_tflops": estimated_tflops,
    "model_path": results.model_path
})

# Log global best parameters
for key, value in results.global_best_params.items():
    wandb.log({f"global_best_params_{key}": value})


# Log global best delta parameters
for key, value in results.global_best_delta_params.items():
    wandb.log({f"global_best_delta_params_{key}": value})


# save as DF
data = {
    "global_max_test_peak": [results.global_max_test_peak],
    "global_best_params": [results.global_best_params],
    "global_delta_test_peak": [results.global_delta_test_peak],
    "global_best_delta_params": [results.global_best_delta_params],
    "model_path": [results.model_path],
    "training_time": [elapsed_time],
    "estimated_tflops": [estimated_tflops]
}

df = pd.DataFrame(data)
print(df)
print(f"Trained convex 2 layer model saved at: {results.model_path}")
print(f"Stage 1 Training completed in {elapsed_time:.2f} seconds")
print(f"Estimated TFLOPS from Stage 1: {estimated_tflops:.2f}")

# Save results for bash (double saved in defrun)
log_out_path = os.path.join(args.output_dir, "cronos_results.txt")
with open(log_out_path, "w") as f:
    for k, v in data.items():
        f.write(f"{k}: {v[0]}\n")


wandb.finish()