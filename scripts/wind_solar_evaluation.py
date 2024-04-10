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

letter_map = {
    "A": [],
    "B": [
        "direct_radiation_24h_ahead_5p",
        "diffuse_radiation_24h_ahead_5p",
        "windspeed_24h_ahead_5p",
    ],
    "C": [
        "direct_radiation_24h_ahead_10p",
        "diffuse_radiation_24h_ahead_10p",
        "windspeed_24h_ahead_10p",
    ],
    "D": [
        "direct_radiation_24h_ahead_20p",
        "diffuse_radiation_24h_ahead_20p",
        "windspeed_24h_ahead_20p",
    ],
    "E": [
        "direct_radiation_24h_ahead_50p",
        "diffuse_radiation_24h_ahead_50p",
        "windspeed_24h_ahead_50p",
    ],
    "F": [
        "direct_radiation_24h_ahead_100p",
        "diffuse_radiation_24h_ahead_100p",
        "windspeed_24h_ahead_100p",
    ],
}

config["dropout"] = 0.061
config["lr"] = 2.9e-04
config["batch_size"] = 32
config["d_model"] = 256
config["N"] = 4
config["enc_seq_len"] = 72
config["output_sequence_length"] = 24
config["step_size"] = 24
config["num_epochs"] = 150
config["predicted_features"] = ["load", "wind", "solar"]
# config["predicted_features"] = ["wind", "solar"]
config["continuous_vars"] = [
    "temperature",
    "cloudcover",
    "windspeed",
    "direct_radiation",
    "diffuse_radiation",
]
config["discrete_vars"] = [
    "region",
    "year",
    "month",
    "day_of_week",
    "day",
    "hour",
]
config["data_path"] = "/scratch/gpfs/awiteck/data/ercot_all_regions_test.csv"


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--letter", type=str, help="Letter of the experiment")

# Parse the arguments
args = parser.parse_args()

if args.letter:
    config["continuous_vars"] = config["continuous_vars"] + letter_map[args.letter]
else:
    print("No model number provided.")
config["experiment_name"] = f"wind_solar_{args.letter}_validation"

regions_list = [
    "North_Central",
    "North",
    "South_Central",
    "East",
    "West",
    "Coast",
    "Far_West",
    "South",
]
for region in regions_list:
    # Load dataset
    print(f"Retrieving data from {config['data_path']}")
    df = pd.read_csv(config["data_path"])
    print(f"Data retrieved.")

    train_data = pd.read_csv("/scratch/gpfs/awiteck/data/ercot_all_regions_train.csv")
    # Ensure all continuous vars are normalized
    train_data = utils.normalize_continuous_vars(
        data=train_data, var_names=config["continuous_vars"]
    )
    df = pd.concat([train_data, df], ignore_index=True)

    data = df[df["region"] == region]

    data["time"] = pd.to_datetime(data["time"])

    # Calculate "2018-12-01" minus 72 hours
    target_date = pd.to_datetime("2018-12-01") - pd.Timedelta(
        hours=config["enc_seq_len"]
    )

    # Select rows on or after this date
    filtered_df = data[data["time"] >= target_date]

    data = filtered_df

    # Ensure all continuous vars are normalized
    data = utils.normalize_continuous_vars(
        data=data, var_names=config["continuous_vars"]
    )
    print("Continuous data normalized.")

    # Ensure all continuous vars are normalized
    data = utils.normalize_discrete_vars(data=data, var_names=config["discrete_vars"])
    print("Discrete data normalized.")

    # Calculate discrete variable dimensions and corresponding embedding dimensions
    discrete_var_dims = utils.calculate_discrete_dims(
        train_data, config["discrete_vars"]
    )

    print(f"discrete_vars: {config['discrete_vars']}")
    print(f"discrete_var_dims: {discrete_var_dims}")

    config["discrete_var_dims"] = discrete_var_dims
    config["discrete_embedding_dims"] = [
        round(1.6 * math.sqrt(dim)) for dim in discrete_var_dims
    ]

    print(f"len(data): {len(data)}")

    # Randomly split train and test indices
    indices = utils.get_indices_entire_sequence(
        data=data,
        window_size=config["enc_seq_len"] + config["output_sequence_length"],
        step_size=24,
    )

    input_variables = (
        config["predicted_features"]
        + config["continuous_vars"]
        + config["discrete_vars"]
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
        config["output_dir"] + "wind_solar_" + args.letter + "_2024_03_09.pt"
    )
    model.load_state_dict(torch.load(model_filename))
    print("Model loaded")

    vals = run_validation(
        model,
        val_data,
        config["output_sequence_length"],
        device,
        num_predicted_features=len(config["predicted_features"]),
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
    output_path = (
        config["output_dir"]
        + config["experiment_name"]
        + f"_{region}"
        + "_example_outputs.json"
    )

    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=4)
