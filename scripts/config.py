from pathlib import Path


def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 150,
        "patience": 15,
        "lr": 0.003,
        "loss_fn": "mse",
        "enc_seq_len": 24,
        "output_sequence_length": 24,
        "d_model": 64,
        "data_path": "/scratch/network/awiteck/data/composite.csv",
        "output_dir": "/scratch/network/awiteck/output/",
        "test_size": 0.1,
        "N": 2,
        "d_ff": 512,
        "dropout": 0.05,
        "step_size": 1,
        "target_col_name": "Normalized Demand",
        "continuous_vars": (["temperature"]),
        "discrete_vars": (["hour", "region_id"]),
        # "preload": "latest",
        # "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "2024_2_20_outputtest",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
