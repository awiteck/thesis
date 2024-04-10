import optuna
from optuna.trial import TrialState
from config import get_config
from basic_train_helper import run_training_and_validation
import pandas as pd
import argparse
import torch.nn as nn
from basic_train_helper import (
    run_training_and_validation,
)
import dataset
from torch.utils.data import DataLoader
import json


def objective(trial, config):
    # Define the range of hyperparameters
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    config["batch_size"] = batch_size
    config["lr"] = lr

    # Train and evaluate the model
    # performance_metric = run_training_and_validation(
    #     config,
    #     save_model=False,
    #     return_vals=True,
    #     trial=trial,
    #     small_dataset=False,
    #     normalize=False,
    # )

    # Train and evaluate the model
    performance_metric = run_training_and_validation(
        config, save_model=False, return_vals=True, trial=trial, small_dataset=False
    )

    return performance_metric["avg_rmse"]


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--type", type=str, help="random or tpe")
args = parser.parse_args()
assert args.type in ("random", "tpe")

config = get_config()

config["num_epochs"] = 15
config["enc_seq_len"] = 24
config["step_size"] = 1
config["output_sequence_length"] = 24
config["loss_fn"] = "huber"
config["data_path"] = "/scratch/gpfs/awiteck/data/composite.csv"
config["experiment_name"] = f"basic_hyperparam_{args.type}"

sampler = (
    optuna.samplers.RandomSampler()
    if args.type == "random"
    else optuna.samplers.TPESampler()
)
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2, interval_steps=1
        ),
        patience=1,
    ),
    sampler=sampler,
)
study.optimize(lambda trial: objective(trial, config), n_trials=50)

# Find number of pruned and completed trials
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


# After the optimization is done
best_params = study.best_params

# Convert the best_params dict to a string for writing to file
best_params_str = "\n".join([f"{key}: {value}" for key, value in best_params.items()])

# Write the best hyperparameters to a file
with open(f"basic_best_params_{args.type}.txt", "w") as f:
    f.write(best_params_str)

print("Best hyperparameters: ", study.best_params)
print(f"Best hyperparameters saved to basic_best_params_{args.type}.txt")
