'''
This file defines the run function for training the convex neural network
it create the cvx_relu_mlp object with given parameters
uses CRONOS and ADMM optimization. It loads data, trains the model, and saves results.
uses 10 neurons 
'''

import jax
import numpy as np
import jax.numpy as jnp
from solve.utils.whisper_dataloader import load_data
from solve.models.cvx_relu_mlp import CVX_ReLU_MLP
from solve.optimizers.admm import admm
from solve.experiments.lr_experiment import lr_random_search

import os
import pickle
import pandas as pd
import time
from typing import NamedTuple


# NamedTuple to return immutable, structured results
class RunResults(NamedTuple):
    global_max_test_peak: float
    global_best_params: dict
    global_delta_test_peak: float
    global_best_delta_params: dict
    model_path: str


def run(model_name, data_dir, cronos_params, adamW_params, opt_seed, data_seed, output_dir, target_lang):
    """
    Run the CRONOS training pipeline for CVX-DPO
    
    Args:
        model_name: Name of the model/dataset
        cronos_params: Parameters for CRONOS optimizer
        adamW_params: Parameters for AdamW optimizer
        opt_seed: Random seed for optimization
        data_seed: Random seed for data loading
        output_dir: Directory to save outputs
    
    Returns:
        RunResults: NamedTuple with results and paths
    """
    global_max_test_peak = 0
    global_best_params = {}  # params that lead to highest CRONOS test peak
    global_delta_test_peak = 0
    global_best_delta_params = {}

    # Load the training and test data
    Atr, ytr = load_data(data_dir, target_lang, data_seed=data_seed, caller_script="defrun", dataset_split="train")
    Atst, ytst = load_data(data_dir, target_lang, data_seed=data_seed, caller_script="defrun", dataset_split="valid")
    # Atr, ytr, Atst, ytst, ntr, ntst = load_data(data_dir, target_lang, data_seed=data_seed, caller_script="defrun")

    ##### CRONOS #####
    # Number of neurons in the convex network 
    num_neurons = cronos_params.get('P_S', 10)
    
    # Create the convex neural network model
    model = CVX_ReLU_MLP(Atr, ytr, num_neurons, cronos_params['beta'], cronos_params['rho'], jax.random.PRNGKey(0))
    model.init_model()
    model.Xtst = Atst
    model.ytst = ytst

    print('Training model with CRONOS')
    
    # Start timing CRONOS training
    cronos_start_time = time.time()
    
    # Run twice to get compiled version 
    for i in range(2):
        (u1, u2), metrics = admm(model, cronos_params)
        if i == 1:
            # End timing after the actual training run (not compilation)
            cronos_end_time = time.time()
            cronos_training_time = cronos_end_time - cronos_start_time
            print('Finished training with CRONOS')
            print(f"CRONOS training time: {cronos_training_time:.2f} seconds")

    # Get peak accuracies
    train_peak = np.max(metrics['train_acc'])
    test_peak = np.max(metrics['val_acc'])
    print(f"Peak train accuracy: {train_peak}")
    print(f"Peak test accuracy: {test_peak}")

    # Update global best if this run is better
    if test_peak > global_max_test_peak:
        global_max_test_peak = test_peak
        print(f"New global max test peak for CXV: {global_max_test_peak}")
        global_best_params = {
            "model_name": model_name,
            "cronos_params": cronos_params,
            "adamW_params": adamW_params,
            "opt_seed": opt_seed,
            "data_seed": data_seed,
            "test_peak": test_peak,
            "train_peak": train_peak
        }

    ##### AdamW #####
    # Compare with AdamW optimizer
    seed_offset = 10
    seeds = [opt_seed, opt_seed + seed_offset, opt_seed + 1 + seed_offset]
    problem_data = dict(training_X=Atr, training_y=ytr, test_X=Atst, test_y=ytst)

    # Start timing AdamW training (all 3 seeds)
    adamw_start_time = time.time()
    adamw_individual_times = []

    for i, seed in enumerate(seeds):
        filename = f"{model_name}_rho_{cronos_params['rho']}_admm_{cronos_params['admm_iters']}_pcg_{cronos_params['pcg_iters']}_seed_{seed}.pkl"
        optimizer_metrics = {'CRONOS': metrics}

        # Parameters for random search
        l, u = -6, -2.5
        grid_size = 8
        tuning_seed = 0

        model_params = dict(type='two_layer_mlp')
        task = 'classification'
        adamW_params['seed'] = jax.random.PRNGKey(seed)

        # Time individual AdamW run
        adamw_seed_start = time.time()
        
        # Run AdamW optimization
        optimizer_metrics['AdamW'] = lr_random_search(problem_data, model_params, adamW_params, task, l, u, grid_size, tuning_seed)
        
        adamw_seed_end = time.time()
        adamw_seed_time = adamw_seed_end - adamw_seed_start
        adamw_individual_times.append(adamw_seed_time)

        print(np.max(optimizer_metrics['AdamW']['test_acc']))
        print(f"Finished running AdamW for seed_{seed}! Time: {adamw_seed_time:.2f} seconds")

        # Calculate improvement of CRONOS over AdamW
        delta_test_peak = test_peak - np.max(optimizer_metrics['AdamW']['test_acc'])
        if delta_test_peak > global_delta_test_peak:
            global_delta_test_peak = delta_test_peak
            print(f"New global delta peak for CXV-AdamW delta: {global_delta_test_peak}")
            global_best_delta_params = {
                "model_name": model_name,
                "cronos_params": cronos_params,
                "adamW_params": adamW_params,
                "test_peak": test_peak,
                "train_peak": train_peak
            }

        # Create the subfolder path
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save optimizer metrics
        pickle_file_path = os.path.join(model_dir, filename)
        with open(pickle_file_path, 'wb') as handle:
            pickle.dump(optimizer_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # End timing AdamW training (all 3 seeds)
    adamw_end_time = time.time()
    adamw_total_time = adamw_end_time - adamw_start_time
    adamw_avg_time = np.mean(adamw_individual_times)

    # Print timing comparison
    print("\n" + "="*50)
    print("TRAINING TIME COMPARISON:")
    print(f"CRONOS training time: {cronos_training_time:.2f} seconds")
    print(f"AdamW total time (3 seeds): {adamw_total_time:.2f} seconds")
    print(f"AdamW average per seed: {adamw_avg_time:.2f} seconds")
    print(f"AdamW individual times: {[f'{t:.2f}s' for t in adamw_individual_times]}")
    print(f"CRONOS vs AdamW (single seed): {cronos_training_time/adamw_avg_time:.2f}x ratio")
    print("="*50 + "\n")

    # Save global metrics CSV
    metrics_df = pd.DataFrame({
        "global_max_test_peak": [global_max_test_peak],
        "global_best_params": [global_best_params],
        "global_delta_test_peak": [global_delta_test_peak],
        "global_best_delta_params": [global_best_delta_params]
    })

    print("Saving CSV to:", os.path.join(output_dir, "global_metrics.csv"))
    metrics_df.to_csv(os.path.join(output_dir, "global_metrics.csv"), sep='\t', encoding='utf-8', index=False, header=True)

    # Save the trained convex model
    trained_model_path = os.path.join(output_dir, f"{model_name}_trained_cvx_mlp.pkl")
    with open(trained_model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Trained convex MLP model saved at: {trained_model_path}")

    # Return results as NamedTuple to cronos_trainer.py
    return RunResults(
        global_max_test_peak=global_max_test_peak,
        global_best_params=global_best_params,
        global_delta_test_peak=global_delta_test_peak,
        global_best_delta_params=global_best_delta_params,
        model_path=trained_model_path
    )



# '''
# This file defines the run function for training the convex neural network
# it create the cvx_relu_mlp object with given parameters
# uses CRONOS and ADMM optimization. It loads data, trains the model, and saves results.
# uses 10 neurons 
# '''

# import jax
# import numpy as np
# import jax.numpy as jnp
# from solve.utils.gpt2_dataloader import load_data
# from solve.models.cvx_relu_mlp import CVX_ReLU_MLP
# from solve.optimizers.admm import admm
# from solve.experiments.lr_experiment import lr_random_search

# import os
# import pickle
# import pandas as pd
# from typing import NamedTuple


# # NamedTuple to return immutable, structured results
# class RunResults(NamedTuple):
#     global_max_test_peak: float
#     global_best_params: dict
#     global_delta_test_peak: float
#     global_best_delta_params: dict
#     model_path: str


# def run(model_name, cronos_params, adamW_params, opt_seed, data_seed, output_dir):
#     """
#     Run the CRONOS training pipeline for CVX-DPO
    
#     Args:
#         model_name: Name of the model/dataset
#         cronos_params: Parameters for CRONOS optimizer
#         adamW_params: Parameters for AdamW optimizer
#         opt_seed: Random seed for optimization
#         data_seed: Random seed for data loading
#         output_dir: Directory to save outputs
    
#     Returns:
#         RunResults: NamedTuple with results and paths
#     """
#     global_max_test_peak = 0
#     global_best_params = {}  # params that lead to highest CRONOS test peak
#     global_delta_test_peak = 0
#     global_best_delta_params = {}

#     # Load the training and test data
#     #Atr, ytr, Atst, ytst, ntr, ntst = load_data(model_name, data_seed)
#     Atr, ytr, Atst, ytst, ntr, ntst = load_data(model_name, data_seed, caller_script="defrun")



#     ##### CRONOS #####
#     # Number of neurons in the convex network 
#     num_neurons = cronos_params.get('P_S', 10)
    
#     # Create the convex neural network model
#     model = CVX_ReLU_MLP(Atr, ytr, num_neurons, cronos_params['beta'], cronos_params['rho'], jax.random.PRNGKey(0))
#     model.init_model()
#     model.Xtst = Atst
#     model.ytst = ytst

#     print('Training model with CRONOS')
#     # Run twice to get compiled version 
#     for i in range(2):
#         (u1, u2), metrics = admm(model, cronos_params)
#         if i == 1:
#             print('Finished training with CRONOS')

#     # Get peak accuracies
#     train_peak = np.max(metrics['train_acc'])
#     test_peak = np.max(metrics['val_acc'])
#     print(f"Peak train accuracy: {train_peak}")
#     print(f"Peak test accuracy: {test_peak}")

#     # Update global best if this run is better
#     if test_peak > global_max_test_peak:
#         global_max_test_peak = test_peak
#         print(f"New global max test peak for CXV: {global_max_test_peak}")
#         global_best_params = {
#             "model_name": model_name,
#             "cronos_params": cronos_params,
#             "adamW_params": adamW_params,
#             "opt_seed": opt_seed,
#             "data_seed": data_seed,
#             "test_peak": test_peak,
#             "train_peak": train_peak
#         }

#     ##### AdamW #####
#     # Compare with AdamW optimizer
#     seed_offset = 10
#     seeds = [opt_seed, opt_seed + seed_offset, opt_seed + 1 + seed_offset]
#     problem_data = dict(training_X=Atr, training_y=ytr, test_X=Atst, test_y=ytst)

#     for seed in seeds:
#         filename = f"{model_name}_rho_{cronos_params['rho']}_admm_{cronos_params['admm_iters']}_pcg_{cronos_params['pcg_iters']}_seed_{seed}.pkl"
#         optimizer_metrics = {'CRONOS': metrics}

#         # Parameters for random search
#         l, u = -6, -2.5
#         grid_size = 8
#         tuning_seed = 0

#         model_params = dict(type='two_layer_mlp')
#         task = 'classification'
#         adamW_params['seed'] = jax.random.PRNGKey(seed)

#         # Run AdamW optimization
#         optimizer_metrics['AdamW'] = lr_random_search(problem_data, model_params, adamW_params, task, l, u, grid_size, tuning_seed)

#         print(np.max(optimizer_metrics['AdamW']['test_acc']))
#         print(f"Finished running AdamW for seed_{seed}!")

#         # Calculate improvement of CRONOS over AdamW
#         delta_test_peak = test_peak - np.max(optimizer_metrics['AdamW']['test_acc'])
#         if delta_test_peak > global_delta_test_peak:
#             global_delta_test_peak = delta_test_peak
#             print(f"New global delta peak for CXV-AdamW delta: {global_delta_test_peak}")
#             global_best_delta_params = {
#                 "model_name": model_name,
#                 "cronos_params": cronos_params,
#                 "adamW_params": adamW_params,
#                 "test_peak": test_peak,
#                 "train_peak": train_peak
#             }

#         # Create the subfolder path
#         model_dir = os.path.join(output_dir, model_name)
#         os.makedirs(model_dir, exist_ok=True)

#         # Save optimizer metrics
#         pickle_file_path = os.path.join(model_dir, filename)
#         with open(pickle_file_path, 'wb') as handle:
#             pickle.dump(optimizer_metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     # Save global metrics CSV
#     metrics_df = pd.DataFrame({
#         "global_max_test_peak": [global_max_test_peak],
#         "global_best_params": [global_best_params],
#         "global_delta_test_peak": [global_delta_test_peak],
#         "global_best_delta_params": [global_best_delta_params]
#     })

#     print("Saving CSV to:", os.path.join(output_dir, "global_metrics.csv"))
#     metrics_df.to_csv(os.path.join(output_dir, "global_metrics.csv"), sep='\t', encoding='utf-8', index=False, header=True)

#     # Save the trained convex model
#     trained_model_path = os.path.join(output_dir, f"{model_name}_trained_cvx_mlp.pkl")
#     with open(trained_model_path, 'wb') as f:
#         pickle.dump(model, f)

#     print(f"Trained convex MLP model saved at: {trained_model_path}")

#     # Return results as NamedTuple to cronos_trainer.py
#     return RunResults(
#         global_max_test_peak=global_max_test_peak,
#         global_best_params=global_best_params,
#         global_delta_test_peak=global_delta_test_peak,
#         global_best_delta_params=global_best_delta_params,
#         model_path=trained_model_path
#     )

