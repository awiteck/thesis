import torch
import torch.nn as nn
from config import get_config
from train_helper import get_model, run_validation
import pandas as pd
import utils
import dataset
from torch.utils.data import DataLoader
import math
import argparse
import json

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
device = torch.device(device)
config = get_config()
config["data_path"] = "/scratch/gpfs/awiteck/data/erco_2023_andcomposite.csv"
# config["N"] = 3
# config["d_model"] = 512
# config["lr"] = 7.3e-5
# config["dropout"] = 0.044

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--letter", type=str, help="Letter of the experiment")
parser.add_argument("--number", type=str, help="Number of the experiment")
parser.add_argument("--s", type=str, help="step size of the experiment")


letter_to_length = {
    "A": {
        "input_length": 24,
        "output_length": 24,
        "N": 2,
        "d_model": 64,
        "lr": 4.4e-03,
        "dropout": 0.047,
    },
    "F": {
        "input_length": 168,
        "output_length": 24,
        "N": 4,
        "d_model": 512,
        "lr": 9.37e-05,
        "dropout": 0.014,
    },
    "C": {
        "input_length": 48,
        "output_length": 24,
        "N": 3,
        "d_model": 512,
        "lr": 2.8e-04,
        "dropout": 0.051,
    },
}
number_to_var = {
    "1": {"continuous_vars": [], "discrete_vars": ["Region"]},
    "2": {"continuous_vars": [], "discrete_vars": ["Region", "day_of_week", "hour"]},
    "3": {
        "continuous_vars": [],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "4": {
        "continuous_vars": ["temperature"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "5": {
        "continuous_vars": ["temperature", "humidity"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "6": {
        "continuous_vars": ["temperature", "windspeed"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "7": {
        "continuous_vars": ["temperature", "cloudcover"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "8": {
        "continuous_vars": ["temperature", "humidity", "windspeed"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "9": {
        "continuous_vars": ["temperature", "windspeed", "cloudcover"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "10": {
        "continuous_vars": ["temperature", "humidity", "cloudcover"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "11": {
        "continuous_vars": ["temperature", "humidity", "windspeed", "cloudcover"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
}

# Parse the arguments
args = parser.parse_args()
if args.letter:
    config["enc_seq_len"] = letter_to_length[args.letter]["input_length"]
    config["output_sequence_length"] = letter_to_length[args.letter]["output_length"]
    config["N"] = letter_to_length[args.letter]["N"]
    config["d_model"] = letter_to_length[args.letter]["d_model"]
    config["lr"] = letter_to_length[args.letter]["lr"]
    config["dropout"] = letter_to_length[args.letter]["dropout"]
else:
    print("No model letter provided.")
if args.number:
    config["continuous_vars"] = number_to_var[args.number]["continuous_vars"]
    config["discrete_vars"] = number_to_var[args.number]["discrete_vars"]
else:
    print("No model number provided.")
config["experiment_name"] = f"{args.letter}{args.number}_s{args.s}_erco_2023"

# Load dataset
print(f"Retrieving data from {config['data_path']}")
df = pd.read_csv(config["data_path"])
print(f"Data retrieved.")
df["timestamp"] = pd.to_datetime(df["timestamp"])

data = df

num_rows = (df["year"] == 2023).sum()

# Locate rows where the 'year' column is 2023 and change them to 2022
df.loc[df["year"] == 2023, "year"] = 2022
df.loc[df["Region"] == "erco_2023", "Region"] = "erco"

target_date = pd.to_datetime("2023-01-01") - pd.Timedelta(hours=config["enc_seq_len"])
df.loc[(df["Region"] == "erco") & (df["timestamp"] > target_date), "2023"] = 1

# Ensure all continuous vars are normalized
data = utils.normalize_continuous_vars(data=data, var_names=config["continuous_vars"])
print("Continuous data normalized.")

# Ensure all continuous vars are normalized
data = utils.normalize_discrete_vars(data=data, var_names=config["discrete_vars"])
print("Discrete data normalized.")

# Calculate discrete variable dimensions and corresponding embedding dimensions
discrete_var_dims = utils.calculate_discrete_dims(data, config["discrete_vars"])

print(f"discrete_vars: {config['discrete_vars']}")
print(f"discrete_var_dims: {discrete_var_dims}")


config["discrete_var_dims"] = discrete_var_dims
config["discrete_embedding_dims"] = [
    round(1.6 * math.sqrt(dim)) for dim in discrete_var_dims
]

data_2023 = data[data["2023"] == 1]
# data_2023 = data[(data["Region"] == "erco") & (data["timestamp"] > target_date)]

data = data_2023

# Randomly split train and test indices
indices = utils.get_indices_entire_sequence(
    data=data,
    window_size=config["enc_seq_len"] + config["output_sequence_length"],
    step_size=24,
)

input_variables = (
    [config["target_col_name"]] + config["continuous_vars"] + config["discrete_vars"]
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
model_filename = (
    config["output_dir"] + args.letter + args.number + f"_s{args.s}_2024_04_01.pt"
)
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

# Write the selected data to a file
output_path = config["output_dir"] + config["experiment_name"] + "_example_outputs.json"

with open(output_path, "w") as f:
    json.dump(converted_data, f, indent=4)
