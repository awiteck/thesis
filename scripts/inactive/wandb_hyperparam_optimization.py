import wandb
from config import get_config
from model import build_transformer, Transformer
from train_helper import run_training_and_validation


def build_wandb_config():
    sweep_config = {"method": "random"}
    metric = {"name": "rmse", "goal": "minimize"}
    sweep_config["metric"] = metric
    parameters_dict = {
        "d_model": {"values": [16, 32, 64, 128, 256]},
        "dropout": {
            "distribution": "uniform",
            "min": 0,
            "max": 0.3,
        },
        "lr": {"distribution": "log_uniform", "min": -5, "max": -1},
    }
    sweep_config["parameters"] = parameters_dict
    return sweep_config


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


def wandb_train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        config = wandb.config
        # Define the range of hyperparameters
        d_model = config.d_model
        dropout = config.dropout
        lr = config.lr

    model_config = get_config()

    model_config["N"] = 2
    model_config["d_ff"] = 512
    model_config["d_model"] = d_model
    model_config["dropout"] = dropout
    model_config["lr"] = lr

    # Train and evaluate the model
    performance_metric = run_training_and_validation(
        model_config, save_model=False, return_vals=True
    )
    wandb.log({"rmse": performance_metric["avg_rmse"]})


sweep_config = build_wandb_config()
sweep_id = wandb.sweep(sweep_config, project="transformer-hyperparam-sweep")
wandb.agent(sweep_id, wandb_train, count=10)


# # After the optimization is done
# best_params = study.best_params

# # Convert the best_params dict to a string for writing to file
# best_params_str = "\n".join([f"{key}: {value}" for key, value in best_params.items()])

# # Write the best hyperparameters to a file
# with open("best_params.txt", "w") as f:
#     f.write(best_params_str)

# print("Best hyperparameters: ", study.best_params)
# print("Best hyperparameters saved to best_params.txt")
