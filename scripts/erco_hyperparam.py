import optuna
from optuna.trial import TrialState
from config import get_config
from model import build_transformer, Transformer
from train_helper import run_training_and_validation
import pandas as pd
import argparse
import utils
import math
import torch
import torch.nn as nn
from train_helper import (
    run_training_and_validation,
    ercot_evaluate,
)
import dataset
from torch.utils.data import DataLoader
import json


def objective(trial, config):
    # Define the range of hyperparameters
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    N = trial.suggest_int("N", 1, 4)
    dropout = trial.suggest_float("dropout", 0, 0.3)
    d_ff = trial.suggest_categorical("d_ff", [128, 256, 512, 1024, 2048])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # s= trial.suggest_int("s", 1, 6)
    # enc_seq_len = trial.suggest_categorical("enc_seq_len", [12, 24, 36, 48, 72, 168])

    config["num_epochs"] = 30
    config["d_ff"] = d_ff
    config["d_model"] = d_model
    config["batch_size"] = batch_size
    config["dropout"] = dropout
    config["lr"] = lr
    config["N"] = N
    config["enc_seq_len"] = 24
    config["step_size"] = 3

    # # Train and evaluate the model
    # performance_metric = run_training_and_validation(
    #     config,
    #     save_model=False,
    #     return_vals=True,
    #     trial=trial,
    #     small_dataset=False,
    #     normalize=False,
    # )

    config["data_path"] = "/scratch/gpfs/awiteck/data/erco_hyperparam_train.csv"
    run_training_and_validation(config, write_example_predictions=True, normalize=False)

    print("Training finished. Evaluating on 2022 data...")

    config["data_path"] = "/scratch/gpfs/awiteck/data/erco_hyperparam_test.csv"

    performance_metric = ercot_evaluate(config)

    return performance_metric


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--type", type=str, help="random or tpe")
parser.add_argument("--s", type=int, help="step size")
# Parse the arguments
args = parser.parse_args()
assert args.type in ("random", "tpe")

config = get_config()

config["output_sequence_length"] = 24
# config["data_path"] = "/scratch/gpfs/awiteck/data/erco_train.csv"
# config["step_size"] = 24
config["predicted_features"] = ["Normalized Demand"]
config["continuous_vars"] = [
    "temperature_1",
    "temperature_2",
    "temperature_3",
    "temperature_4",
    "temperature_5",
    "temperature_6",
    "precipitation_1",
    "precipitation_2",
    "precipitation_3",
    "precipitation_4",
    "precipitation_5",
    "precipitation_6",
    "humidity_1",
    "humidity_2",
    "humidity_3",
    "humidity_4",
    "humidity_5",
    "humidity_6",
]
config["discrete_vars"] = ["month", "day_of_week", "hour", "is_holiday"]
config["experiment_name"] = f"erco_hyperparam_{args.type}_s{args.s}"
config["step_size"] = args.s


# study = optuna.create_study(
#     direction="minimize", sampler=optuna.samplers.RandomSampler()
# )
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
with open(f"solar_best_params_{args.type}.txt", "w") as f:
    f.write(best_params_str)

print("Best hyperparameters: ", study.best_params)
print(f"Best hyperparameters saved to solar_best_params_{args.type}.txt")
