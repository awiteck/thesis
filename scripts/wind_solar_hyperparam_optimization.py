import optuna
from optuna.trial import TrialState
from config import get_config
from model import build_transformer, Transformer
from train_helper import run_training_and_validation
import pandas as pd


def objective(trial, config):
    # Define the range of hyperparameters
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512, 1024])
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    N = trial.suggest_int("N", 1, 6)
    dropout = trial.suggest_float("dropout", 0, 0.3)
    # d_ff = trial.suggest_categorical("d_ff", [128, 256, 512, 1024, 2048])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    enc_seq_len = trial.suggest_categorical("enc_seq_len", [12, 24, 36, 48, 72, 168])

    # config["N"] = 2
    config["num_epochs"] = 30
    config["d_ff"] = 512
    config["d_model"] = d_model
    config["batch_size"] = batch_size
    config["dropout"] = dropout
    config["lr"] = lr
    config["N"] = N
    config["enc_seq_len"] = enc_seq_len
    config["step_size"] = enc_seq_len + config["output_sequence_length"]

    # Train and evaluate the model
    performance_metric = run_training_and_validation(
        config, save_model=False, return_vals=True, trial=trial
    )

    return performance_metric["avg_rmse"]


config = get_config()

config["output_sequence_length"] = 24
config["data_path"] = "/scratch/network/awiteck/data/ercot_all_regions_train.csv"
# config["step_size"] = 1
config["predicted_features"] = ["load", "wind", "solar"]
config["continuous_vars"] = [
    "temperature",
    "cloudcover",
    "windspeed",
    "direct_radiation",
    "diffuse_radiation",
    "direct_radiation_24h_ahead_50p",
    "diffuse_radiation_24h_ahead_50p",
    "windspeed_24h_ahead_50p",
]
config["discrete_vars"] = [
    "region",
    "year",
    "month",
    "day_of_week",
    "day",
    "hour",
]
config["experiment_name"] = "wind_solar_random_50p"
# config["experiment_name"] = "wind_solar_tpe_50p_nooverlap"


# study = optuna.create_study(
#     direction="minimize", sampler=optuna.samplers.RandomSampler()
# )
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2, interval_steps=1
        ),
        patience=1,
    ),
    sampler=optuna.samplers.RandomSampler()
    # sampler=optuna.samplers.TPESampler(),
)
study.optimize(lambda trial: objective(trial, config), n_trials=100)

# Find number of pruned and completed trials
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])


# After the optimization is done
best_params = study.best_params

# Convert the best_params dict to a string for writing to file
best_params_str = "\n".join([f"{key}: {value}" for key, value in best_params.items()])


with open("wind_solar_50p_best_params_random.txt", "w") as f:
    f.write(best_params_str)

# with open("wind_solar_50p_best_params_tpe.txt", "w") as f:
#     f.write(best_params_str)

print("Best hyperparameters: ", study.best_params)
# print("Best hyperparameters saved to wind_solar_50p_nooverlap_best_params_tpe.txt")

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
