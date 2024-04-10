from basic_model import BasicLinearNet

# from dataset import BilingualDataset, causal_mask
from config import get_config  # , get_weights_file_path, latest_weights_file_path

# import torchtext.datasets as datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from EarlyStopper import EarlyStopper

from torch.utils.data import DataLoader, random_split
import optuna
import json

# from torch.optim.lr_scheduler import LambdaLR

import pandas as pd
import utils
import dataset
import math
import random

import warnings

# import os

# Huggingface datasets and tokenizers
# from datasets import load_dataset
# from tokenizers import Tokenizer
# from tokenizers.models import WordLevel
# from tokenizers.trainers import WordLevelTrainer
# from tokenizers.pre_tokenizers import Whitespace

# import torchmetrics
# from torch.utils.tensorboard import SummaryWriter


def run_training_and_validation(
    config,
    save_model=True,
    return_vals=False,
    trial=None,
    write_example_predictions=False,
    small_dataset=False,
    normalize=True,
    shuffle=False,
):
    print(
        f"Running training and validation with d_model={config['d_model']}, dropout={config['dropout']}, lr={config['lr']}"
    )
    """
    This function does the following

    (1) Train model for specified number of epochs (in config),
    evaluating the model on a validation dataset for each epoch.
    Early stopping incoroprated

    (2) Evaluate model on test data

    """

    # Define the device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.has_mps or torch.backends.mps.is_available()
        else "cpu"
    )
    print("Using device:", device)
    if device == "cuda":
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(
            f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB"
        )
    elif device == "mps":
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print(
            "      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc"
        )
        print(
            "      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu"
        )
    device = torch.device(device)

    print(f"Retrieving data from {config['data_path']}")
    df = pd.read_csv(config["data_path"])
    print(f"Data retrieved.")

    weather_columns = ["temperature", "cloudcover", "humidity", "windspeed"]

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
    data = data[input_vars].astype("float32")

    # print(data[input_vars].dtypes)

    # Randomly split train and test indices
    indices = utils.get_indices_entire_sequence(
        data=data,
        window_size=config["enc_seq_len"] + config["output_sequence_length"],
        step_size=config["step_size"],
    )
    if small_dataset:
        total_size = len(indices)  # Replace 'dataset' with your actual dataset variable
        split_size = int(total_size * 0.01)  # 1% of the dataset size

        remaining_size = total_size - 50 * split_size

        # Split the dataset
        training_indices, val_indices, test_indices, _ = random_split(
            indices, [40 * split_size, 5 * split_size, 5 * split_size, remaining_size]
        )
    else:
        training_indices, val_indices, test_indices = random_split(
            indices, [0.8, 0.1, 0.1]
        )

    input_variables = input_vars
    print(f"Input variables: {input_variables}")

    # Making instance of custom dataset class
    training_data = dataset.BasicDataset(
        data=torch.tensor(data[input_variables].values).float(),
        indices=training_indices,
        enc_seq_len=config["enc_seq_len"],
        target_seq_len=config["output_sequence_length"],
    )

    val_data = dataset.BasicDataset(
        data=torch.tensor(data[input_variables].values).float(),
        indices=val_indices,
        enc_seq_len=config["enc_seq_len"],
        target_seq_len=config["output_sequence_length"],
    )

    test_data = dataset.BasicDataset(
        data=torch.tensor(data[input_variables].values).float(),
        indices=test_indices,
        enc_seq_len=config["enc_seq_len"],
        target_seq_len=config["output_sequence_length"],
    )

    training_data = DataLoader(training_data, config["batch_size"])
    val_data = DataLoader(val_data, config["batch_size"])
    test_data = DataLoader(test_data, 1)

    model = BasicLinearNet(T=config["enc_seq_len"]).to(device)

    print("Got model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    global_step = 0
    if config["loss_fn"] == "mse":
        loss_fn = nn.MSELoss().to(device)
    elif config["loss_fn"] == "huber":
        loss_fn = nn.HuberLoss().to(device)
    else:
        loss_fn = nn.MSELoss().to(device)

    early_stopper = EarlyStopper(patience=config["patience"], min_delta=0)
    best_val_loss = float("inf")
    best_model_state = None  # To save the best model's state dict

    print("Beginning training...")
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        torch.cuda.empty_cache()
        model.train()

        for batch_idx, batch in enumerate(training_data):
            model_input, label = batch
            model_input = model_input.to(device)
            demand_weather = model_input[:, :, :9]  # shape: (batch_size, T, 9)
            if shuffle:
                demand_weather = shuffle_tensor_along_T(demand_weather)

            # print(f"demand_weather.size(): {demand_weather.size()}")
            calendar_region = model_input[:, -1, 9:]  # shape: (batch_size, 84)
            # print(f"calendar_region.size(): {calendar_region.size()}")
            demand_weather_flat = demand_weather.reshape(model_input.shape[0], -1)
            # print(f"demand_weather_flat.size(): {demand_weather_flat.size()}")
            final_input = torch.cat(
                (demand_weather_flat, calendar_region), dim=1
            )  # shape: (batch_size, T*9 + 84)
            # print(f"final_input.size(): {final_input.size()}")

            label = label[:, :, 0].to(device)
            # print(f"label.size(): {label.size()}")
            outputs = model(final_input)
            # print(f"outputs.size(): {outputs.size()}")
            loss = loss_fn(outputs, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += loss.item()

            last_loss = loss.item()

        val_epoch_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                model_input, label = batch
                model_input = model_input.to(device)
                demand_weather = model_input[:, :, :9]  # shape: (batch_size, T, 9)
                if shuffle:
                    demand_weather = shuffle_tensor_along_T(demand_weather)
                calendar_region = model_input[:, -1, 9:]  # shape: (batch_size, 84)
                demand_weather_flat = demand_weather.reshape(model_input.shape[0], -1)
                final_input = torch.cat(
                    (demand_weather_flat, calendar_region), dim=1
                )  # shape: (batch_size, T*9 + 84)
                label = label[:, :, 0].to(device)
                outputs = model(final_input)
                loss = loss_fn(outputs, label)
                val_epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(training_data)
        avg_val_loss = val_epoch_loss / (batch_idx + 1)

        print(
            f"{config['experiment_name']}: Epoch {epoch+1}/{config['num_epochs']}, Last Batch Loss: {last_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
        )

        if trial:
            trial.report(avg_val_loss, epoch)
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Check if the current validation loss is the best one we've seen so far
        if avg_val_loss < best_val_loss:
            print(
                f"Validation loss decreased ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model ..."
            )
            best_val_loss = avg_val_loss  # Update the best validation loss

            # Save the best model state instead of immediately saving the model
            best_model_state = model.state_dict()

        if early_stopper.early_stop(val_epoch_loss):
            break

    if save_model and best_model_state is not None:
        output_path = config["output_dir"] + config["experiment_name"] + ".pt"
        print(f"Saving model to {output_path}")
        print("...")
        torch.save(best_model_state, output_path)
        print("Model successfully saved.")

    model.load_state_dict(best_model_state)
    vals = run_validation(
        model,
        test_data,
        config["output_sequence_length"],
        device,
        num_predicted_features=len(config["predicted_features"]),
    )

    if write_example_predictions:
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

    if return_vals:
        return vals


def run_validation(
    model,
    validation_data,
    max_len,
    device,
    num_predicted_features=1,
    num_examples=2,
):
    model.eval()
    count = 0

    source_readings = []
    expected = []
    predicted = []

    mse = 0.0
    mape = 0.0
    rmse = 0.0
    mae = 0.0

    console_width = 80

    with torch.no_grad():
        for batch_idx, batch in enumerate(validation_data):
            count += 1

            model_input, label = batch
            model_input = model_input.to(device)
            demand_weather = model_input[:, :, :9]  # shape: (batch_size, T, 9)
            calendar_region = model_input[:, -1, 9:]  # shape: (batch_size, 84)
            demand_weather_flat = demand_weather.reshape(model_input.shape[0], -1)
            final_input = torch.cat(
                (demand_weather_flat, calendar_region), dim=1
            )  # shape: (batch_size, T*9 + 84)
            label = label[:, :, 0].to(device)
            model_out = model(final_input)

            mse += F.mse_loss(model_out, label)
            mape += utils.mape_loss(model_out, label)
            rmse += utils.rmse_loss(model_out, label)
            mae += utils.mae_loss(model_out, label)

            if batch_idx % 1 == 0:
                source_readings.append(model_input)
                expected.append(label)
                predicted.append(model_out)

    print(f"count: {count}")
    print(len(validation_data))
    print(f"MSE: {mse}")
    print(f"MAPE: {mape}")
    print(f"AVERAGE MAPE: {mape/len(validation_data)}")
    print(f"RMSE: {rmse}")
    print(f"AVERAGE RMSE: {rmse/len(validation_data)}")
    print(f"MAE: {mae}")
    print(f"AVERAGE MAE: {mae/len(validation_data)}")

    return {
        "count": count,
        "mse": mse,
        "mape": mape,
        "avg_mape": mape / len(validation_data),
        "rmse": rmse,
        "avg_rmse": rmse / len(validation_data),
        "mae": mae,
        "avg_mae": mae / len(validation_data),
        "source_readings": source_readings,
        "expected": expected,
        "predicted": predicted,
    }


def shuffle_tensor_along_T(tensor):
    _, T, _ = tensor.shape
    # Generate a single set of shuffled indices for dimension T
    shuffled_indices = torch.randperm(T)
    # Apply the same shuffled indices across all batches and input dimensions
    shuffled_tensor = tensor[:, shuffled_indices, :]
    return shuffled_tensor


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    run_training_and_validation(config)
