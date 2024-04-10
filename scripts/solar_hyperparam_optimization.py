import optuna
from optuna.trial import TrialState
from config import get_config
from model import build_transformer, Transformer
from train_helper import run_training_and_validation
import pandas as pd
import argparse


def objective(trial, config):
    # Define the range of hyperparameters
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    N = trial.suggest_int("N", 1, 6)
    dropout = trial.suggest_float("dropout", 0, 0.3)
    # d_ff = trial.suggest_categorical("d_ff", [128, 256, 512, 1024, 2048])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # enc_seq_len = trial.suggest_categorical("enc_seq_len", [12, 24, 36, 48, 72, 168])

    config["num_epochs"] = 30
    config["d_ff"] = 512
    config["d_model"] = d_model
    config["batch_size"] = batch_size
    config["dropout"] = dropout
    config["lr"] = lr
    config["N"] = N
    # config["enc_seq_len"] = enc_seq_len
    config["step_size"] = 48  # enc_seq_len + 24

    # Train and evaluate the model
    performance_metric = run_training_and_validation(
        config, save_model=False, return_vals=True, trial=trial, small_dataset=False
    )

    return performance_metric["avg_rmse"]


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--type", type=str, help="random or tpe")
# Parse the arguments
args = parser.parse_args()
assert args.type in ("random", "tpe")

config = get_config()

config["output_sequence_length"] = 24
config["data_path"] = "/scratch/gpfs/awiteck/data/ERCOT_solar_train.csv"
# config["step_size"] = 24
config["predicted_features"] = ["normalized_generation"]
config["continuous_vars"] = [
    "temperature",
    "cloudcover",
    "windspeed",
    "direct_radiation",
    "diffuse_radiation",
    "direct_radiation_24h_ahead_50p",
    "diffuse_radiation_24h_ahead_50p",
    "windspeed_24h_ahead_50p",
    "temperature_24h_ahead_50p",
]
config["discrete_vars"] = [
    "source",
    "year",
    "month",
    "day_of_week",
    "day",
    "hour",
]
config["experiment_name"] = f"solar_hyperparam_{args.type}_large_step"


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

# Save results to csv file
# df = study.trials_dataframe().drop(
#     ["datetime_start", "datetime_complete", "duration"], axis=1
# )  # Exclude columns
# df = df.loc[df["state"] == "COMPLETE"]  # Keep only results that did not prune
# df = df.drop("state", axis=1)  # Exclude state column
# df = df.sort_values("value")  # Sort based on accuracy
# df.to_csv("optuna_results_bayesian_full_168.csv", index=False)  # Save to csv file

# # Find the most important hyperparameters
# most_important_parameters = optuna.importance.get_param_importances(study, target=None)
# most_important_parameters_str = "\n".join(
#     [
#         "  {}:{}{:.2f}%".format(key, (15 - len(key)) * " ", value * 100)
#         for key, value in most_important_parameters.items()
#     ]
# )
# # Write the best hyperparameters to a file
# with open("most_important_parameters_bayesian_full_168.txt", "w") as f:
#     f.write(most_important_parameters_str)
