import torch
import torch.nn as nn
from config import get_config
from basic_train_helper import run_validation
import pandas as pd
import utils
import dataset
from torch.utils.data import DataLoader
import math
import argparse
import json
from basic_model import BasicLinearNet


def test_2023(experiment_name, T=24):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    device = torch.device(device)
    config = get_config()
    config["enc_seq_len"] = T
    config["data_path"] = "/scratch/gpfs/awiteck/data/erco_2023_andcomposite.csv"

    config["experiment_name"] = experiment_name

    # Load dataset
    print(f"Retrieving data from {config['data_path']}")
    df = pd.read_csv(config["data_path"])
    print(f"Data retrieved.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[df["year"] <= 2023]

    # Locate rows where the 'year' column is 2023 and change them to 2022
    df.loc[df["year"] == 2023, "year"] = 2022
    df.loc[df["Region"] == "erco_2023", "Region"] = "erco"

    target_date = pd.to_datetime("2023-01-01") - pd.Timedelta(hours=T)
    df["2023"] = 0
    df.loc[(df["Region"] == "erco") & (df["timestamp"] > target_date), "2023"] = 1

    weather_columns = ["temperature", "cloudcover", "humidity", "windspeed"]

    df["temperature"] = (df["temperature"] - 15.411) / 10.824
    df["cloudcover"] = (df["cloudcover"] - 35.8275) / 37.618
    df["humidity"] = (df["humidity"] - 65.774) / 22.957
    df["windspeed"] = (df["windspeed"] - 11.787) / 6.202

    # Ensure all continuous vars are normalized
    df = utils.normalize_continuous_vars(
        data=df, var_names=["temperature", "cloudcover", "humidity", "windspeed"]
    )
    print("Continuous data normalized.")

    for col in weather_columns:
        df[f"{col}_24h_ahead"] = df.groupby("Region")[col].shift(-24)

    df = df.dropna(
        subset=[
            "windspeed_24h_ahead",
            "temperature_24h_ahead",
            "cloudcover_24h_ahead",
            "humidity_24h_ahead",
        ]
    )

    # One-hot encode the 'month', 'day_of_week', 'hour', and 'Region' columns
    categorical_columns = ["year", "month", "day", "day_of_week", "hour", "Region"]
    # Step 1: Before encoding, retrieve the unique values for each categorical variable
    unique_values_per_categorical_column = {
        col: df[col].unique() for col in categorical_columns
    }
    df = pd.get_dummies(df, columns=categorical_columns)

    # Step 3: Construct the expected new column names for each categorical variable
    input_vars = [
        "Normalized Demand",
        "temperature",
        "cloudcover",
        "humidity",
        "windspeed",
        "temperature_24h_ahead",
        "cloudcover_24h_ahead",
        "humidity_24h_ahead",
        "windspeed_24h_ahead",
    ]
    for col, unique_values in unique_values_per_categorical_column.items():
        # Sort the unique values to maintain order; this step may be adjusted based on specific needs
        sorted_unique_values = sorted(unique_values)
        for val in sorted_unique_values:
            new_col_name = f"{col}_{val}"
            if (
                new_col_name in df.columns
            ):  # Ensure the column name exists in the DataFrame
                input_vars.append(new_col_name)

    data = df

    data_2023 = data[data["2023"] == 1][input_vars].astype("float32")
    # data_2023 = data[(data["Region"] == "erco") & (data["timestamp"] > target_date)]

    data = data_2023
    print(f"len(data): {len(data)}")
    print(f"config['enc_seq_len']: {config['enc_seq_len']}")

    # Randomly split train and test indices
    indices = utils.get_indices_entire_sequence(
        data=data,
        window_size=config["enc_seq_len"] + config["output_sequence_length"],
        step_size=24,
    )
    print(len(indices))

    input_variables = input_vars
    print(f"Input variables: {input_variables}")

    # Making instance of custom dataset class
    val_data = dataset.BasicDataset(
        data=torch.tensor(data[input_variables].values).float(),
        indices=indices,
        enc_seq_len=config["enc_seq_len"],
        target_seq_len=config["output_sequence_length"],
    )
    val_data = DataLoader(val_data, 1)

    model = BasicLinearNet(T=T).to(device)

    # Load the pretrained weights
    print("Loading model")
    model_filename = config["output_dir"] + f"{config['experiment_name']}.pt"
    model.load_state_dict(torch.load(model_filename))
    print("Model loaded")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

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
    output_path = (
        config["output_dir"] + config["experiment_name"] + "_example_outputs.json"
    )

    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=4)
