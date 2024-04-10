from config import get_config
import argparse
import pandas as pd
import utils
import math
import torch
import torch.nn as nn
from train_helper import get_model, run_validation, run_training_and_validation
import dataset
from torch.utils.data import DataLoader
import json

config = get_config()


config["num_epochs"] = 100

config["data_path"] = "/scratch/gpfs/awiteck/data/erco_train.csv"
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


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--dropout", type=float, help="dropout")
parser.add_argument("--lr", type=float, help="lr")
parser.add_argument("--batch", type=int, help="batch")
parser.add_argument("--d_model", type=int, help="d_model")
parser.add_argument("--N", type=int, help="N")
parser.add_argument("--T", type=int, help="T")
parser.add_argument("--s", type=int, help="s")
parser.add_argument("--d_ff", type=int, help="d_ff")

# Parse the arguments
args = parser.parse_args()


config["dropout"] = args.dropout  # 0.061
config["lr"] = args.lr  # 2.9e-04
config["batch_size"] = args.batch
config["d_model"] = args.d_model  # 256
config["N"] = args.N  # 4
config["enc_seq_len"] = args.T  # 72
config["output_sequence_length"] = 24
config["step_size"] = args.s  # 12 + 24  # 1
config["d_ff"] = args.d_ff

config["experiment_name"] = f"ercot_s{args.s}_T{args.T}_N{args.N}"


run_training_and_validation(config, write_example_predictions=True, normalize=False)


print("Training finished. Evaluating on 2023 data...")


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
device = torch.device(device)
config["data_path"] = "/scratch/gpfs/awiteck/data/erco_test.csv"

# Load dataset
print(f"Retrieving data from {config['data_path']}")
df = pd.read_csv(config["data_path"])
print(f"Data retrieved.")
df["timestamp"] = pd.to_datetime(df["timestamp"])

data = df

# Calculate discrete variable dimensions and corresponding embedding dimensions
print(f"TEST: {config['discrete_var_dims']}")
# discrete_var_dims = utils.calculate_discrete_dims(data, config["discrete_vars"])

print(f"discrete_vars: {config['discrete_vars']}")
print(f"discrete_var_dims: {config['discrete_var_dims']}")


# config["discrete_var_dims"] = config["discrete_var_dims"]
config["discrete_embedding_dims"] = [
    round(1.6 * math.sqrt(dim)) for dim in config["discrete_var_dims"]
]

# Randomly split train and test indices
indices = utils.get_indices_entire_sequence(
    data=data,
    window_size=config["enc_seq_len"] + config["output_sequence_length"],
    step_size=24,
)

input_variables = (
    config["predicted_features"] + config["continuous_vars"] + config["discrete_vars"]
)

print(f"Input variables: {input_variables}")

# Making instance of custom dataset class
val_data = dataset.TransformerDataset(
    data=torch.tensor(data[input_variables].values).float(),
    indices=indices,
    enc_seq_len=config["enc_seq_len"],
    target_seq_len=config["output_sequence_length"],
)
val_data = DataLoader(val_data, 1)
encoder_mask = None

decoder_mask = utils.causal_mask(config["output_sequence_length"]).to(device)

model = get_model(config).to(device)

# Load the pretrained weights
print("Loading model")
model_filename = config["output_dir"] + config["experiment_name"] + ".pt"
model.load_state_dict(torch.load(model_filename))
print("Model loaded")

vals = run_validation(
    model,
    val_data,
    config["output_sequence_length"],
    device,
)
# Keys of interest
keys_of_interest = [
    "source_readings",
    "expected",
    "predicted",
]
# Convert only the specified keys
converted_data = {
    key: [tensor.tolist() for tensor in vals[key]] for key in keys_of_interest
}
converted_data["avg_mape"] = vals["avg_mape"].item()
converted_data["avg_rmse"] = vals["avg_rmse"].item()
converted_data["avg_mae"] = vals["avg_mae"].item()

# Write the selected data to a file
output_path = config["output_dir"] + config["experiment_name"] + "_example_outputs.json"

with open(output_path, "w") as f:
    json.dump(converted_data, f, indent=4)
