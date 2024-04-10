import optuna
from optuna.trial import TrialState
from config import get_config
from model import build_transformer, Transformer
from train_helper import run_training_and_validation
import pandas as pd


def get_model(config):
    model = build_transformer(
        1 + len(config["continuous_vars"]) + sum(config["discrete_embedding_dims"]),
        config["discrete_var_dims"],
        config["discrete_embedding_dims"],
        config["enc_seq_len"],
        config["output_sequence_length"],
        num_predicted_features=1,
        d_model=config["d_model"],
        N=config["N"],
        d_ff=config["d_ff"],
    )
    return model


def objective(trial, config):
    # Define the range of hyperparameters
    d_model = trial.suggest_categorical("d_model", [16, 32, 64, 128, 256, 512, 1024])
    N = trial.suggest_int("N", 1, 6)
    dropout = trial.suggest_float("dropout", 0, 0.3)
    # d_ff = trial.suggest_categorical("d_ff", [128, 256, 512, 1024, 2048])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # enc_seq_len = trial.suggest_categorical("enc_seq_len", [12, 24, 36, 48, 72, 168])

    # config["N"] = 2
    config["num_epochs"] = 1
    config["d_ff"] = 512
    config["d_model"] = d_model
    config["dropout"] = dropout
    config["lr"] = lr
    config["N"] = N
    config["enc_seq_len"] = 168

    # Train and evaluate the model
    performance_metric = run_training_and_validation(
        config, save_model=False, return_vals=True, trial=trial, small_dataset=False
    )

    return performance_metric["avg_rmse"]


config = get_config()

config["step_size"] = 24
config["output_sequence_length"] = 24
config["data_path"] = "/scratch/network/awiteck/data/composite.csv"
config["predicted_features"] = ["Normalized Demand"]
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
    # sampler=optuna.samplers.RandomSampler()
    sampler=optuna.samplers.TPESampler(),
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
with open("best_params_bayesian_full_168.txt", "w") as f:
    f.write(best_params_str)

print("Best hyperparameters: ", study.best_params)
print("Best hyperparameters saved to best_params_bayesian_full_168.txt")

# Save results to csv file
df = study.trials_dataframe().drop(
    ["datetime_start", "datetime_complete", "duration"], axis=1
)  # Exclude columns
df = df.loc[df["state"] == "COMPLETE"]  # Keep only results that did not prune
df = df.drop("state", axis=1)  # Exclude state column
df = df.sort_values("value")  # Sort based on accuracy
df.to_csv("optuna_results_bayesian_full_168.csv", index=False)  # Save to csv file

# Find the most important hyperparameters
most_important_parameters = optuna.importance.get_param_importances(study, target=None)
most_important_parameters_str = "\n".join(
    [
        "  {}:{}{:.2f}%".format(key, (15 - len(key)) * " ", value * 100)
        for key, value in most_important_parameters.items()
    ]
)
# Write the best hyperparameters to a file
with open("most_important_parameters_bayesian_full_168.txt", "w") as f:
    f.write(most_important_parameters_str)
