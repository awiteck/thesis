from model import build_transformer

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
    print(f"Predicted features: {config['predicted_features']}")
    print(f"Continuous vars: {config['continuous_vars']}")
    print(f"Discrete vars: {config['discrete_vars']}")

    print(f"Retrieving data from {config['data_path']}")
    df = pd.read_csv(config["data_path"])
    print(f"Data retrieved.")
    data = df

    # Ensure all continuous vars are normalized
    data = utils.normalize_continuous_vars(
        data=data, var_names=config["continuous_vars"]
    )
    print("Continuous data normalized.")

    # Ensure all continuous vars are normalized
    data = utils.normalize_discrete_vars(data=data, var_names=config["discrete_vars"])
    print("Discrete data normalized.")

    # Calculate discrete variable dimensions and corresponding embedding dimensions
    discrete_var_dims = utils.calculate_discrete_dims(data, config["discrete_vars"])

    print(f"discrete_var_dims: {discrete_var_dims}")
    config["discrete_var_dims"] = discrete_var_dims
    config["discrete_embedding_dims"] = [
        round(1.6 * math.sqrt(dim)) for dim in discrete_var_dims
    ]

    # Randomly split train and test indices
    indices = utils.get_indices_entire_sequence(
        data=data,
        window_size=config["enc_seq_len"] + config["output_sequence_length"],
        step_size=config["step_size"],
    )

    train_indices, val_indices, test_indices = random_split(indices, [0.8, 0.1, 0.1])

    # # Convert to sets for easier manipulation
    # train_indices_set = set(training_indices)
    # val_indices_set = set(val_indices)
    # test_indices_set = set(test_indices)

    print(f"len(train_indices): {len(train_indices)}")
    print(f"len(val_indices): {len(val_indices)}")
    print(f"len(test_indices): {len(test_indices)}")

    # Remove test set overlaps from training and validation sets
    train_indices = utils.remove_overlapping_intervals(
        test_indices,
        train_indices,
        # config["enc_seq_len"] + config["output_sequence_length"],
    )
    val_indices = utils.remove_overlapping_intervals(
        test_indices,
        val_indices,
        # config["enc_seq_len"] + config["output_sequence_length"],
    )

    # Optionally, remove validation set overlaps from the training set if needed
    train_indices = utils.remove_overlapping_intervals(
        val_indices,
        train_indices,
        # config["enc_seq_len"] + config["output_sequence_length"],
    )

    # Convert back to lists or numpy arrays if necessary
    # train_indices = list(train_indices_set)
    # val_indices = list(val_indices_set)
    # test_indices = list(test_indices_set)

    print(f"len(train_indices): {len(train_indices)}")
    print(f"len(val_indices): {len(val_indices)}")
    print(f"len(test_indices): {len(test_indices)}")

    input_variables = (
        config["predicted_features"]
        + config["continuous_vars"]
        + config["discrete_vars"]
    )

    print(f"Input variables: {input_variables}")

    # Making instance of custom dataset class
    training_data = dataset.TransformerDataset(
        data=torch.tensor(data[input_variables].values).float(),
        indices=train_indices,
        enc_seq_len=config["enc_seq_len"],
        target_seq_len=config["output_sequence_length"],
    )
    val_data = dataset.TransformerDataset(
        data=torch.tensor(data[input_variables].values).float(),
        indices=val_indices,
        enc_seq_len=config["enc_seq_len"],
        target_seq_len=config["output_sequence_length"],
    )

    test_data = dataset.TransformerDataset(
        data=torch.tensor(data[input_variables].values).float(),
        indices=test_indices,
        enc_seq_len=config["enc_seq_len"],
        target_seq_len=config["output_sequence_length"],
    )

    training_data = DataLoader(training_data, config["batch_size"])
    val_data = DataLoader(val_data, config["batch_size"])
    test_data = DataLoader(test_data, 1)
    encoder_mask = None

    decoder_mask = utils.causal_mask(config["output_sequence_length"]).to(device)

    model = get_model(config).to(device)

    print("Got model.")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9
    )
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
            encoder_input, decoder_input, label = batch
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            label = (
                label[:, :, 0 : len(config["predicted_features"])]
                .squeeze(-1)
                .to(device)
            )

            # print(f"encoder input: {encoder_input}")

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)

            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)

            map_output = model.map(decoder_output)

            # Flatten tensors and then compute MSE loss
            loss = loss_fn(map_output.view(-1), label.view(-1))

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += loss.item()

            if batch_idx % 10 == 9:
                last_loss = epoch_loss / 10  # loss per batch
                epoch_loss = 0.0

        val_epoch_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_data):
                encoder_input, decoder_input, label = batch
                encoder_input = encoder_input.to(device)
                decoder_input = decoder_input.to(device)
                label = (
                    label[:, :, 0 : len(config["predicted_features"])]
                    .squeeze(-1)
                    .to(device)
                )

                # Run the tensors through the encoder, decoder and the projection layer
                encoder_output = model.encode(
                    encoder_input, encoder_mask
                )  # (B, seq_len, d_model)

                decoder_output = model.decode(
                    encoder_output, encoder_mask, decoder_input, decoder_mask
                )  # (B, seq_len, d_model)

                map_output = model.map(decoder_output)
                # print(f"output: {map_output.view(-1)}")

                # Flatten tensors and then compute loss
                # print(f"computing loss on {map_output.view(-1)} and {label.view(-1)}")
                loss = loss_fn(map_output.view(-1), label.view(-1))
                # print(f"loss: {loss.item()}")

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


def greedy_decode(
    model,
    source,
    source_mask,
    max_len,
    device,
    input_dim=1,
    num_predicted_features=1,
    contextual_vars=None,
):
    """
    contextual_vars is a 2D tensor of shape (max_len, input_dim-1)
    """
    assert not (
        input_dim > num_predicted_features and contextual_vars is None
    ), "For inputs of more than num_predicted_features features, contextual vars must be given beforehand."

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # print(f"Source size: {source.size()}")
    # decoder_input = (
    #     torch.empty(1, 1, 1).type_as(source).fill_(source[0, -1, 0]).to(device)
    # )
    # Construct initial decoder input
    decoder_input = source[0, -1, :].unsqueeze(0).unsqueeze(0).to(device)

    decoder_output = torch.empty(
        0, num_predicted_features, device=device, dtype=source.dtype
    )
    # decoder_output = torch.empty(0).to(device)

    # decoder_input = torch.empty(1, 1, input_dim).type_as(source).to(device)
    # for i in range(len(source[0, -1, :])):
    #     decoder_input[:, :, i] = source[0, -1, i]

    # for i in range(1, input_dim):
    #     decoder_input[:, :, i] = contextual_vars[0, -1, 0]

    i = 0
    while True:
        # print(f"decoder_input: {decoder_input}")
        # print(f"decoder_input.size(): {decoder_input.size()}")

        # NOTE: Commenting this out as I think I need to be using decoder_output now
        # print(
        #     f"i: {i}, max_len: {max_len}, decoder_input.size(1): {decoder_input.size(1)}"
        # )

        if decoder_input.size(1) == max_len + 1:
            break

        # if decoder_output.size() == max_len:
        #     break

        # build mask for target
        decoder_mask = utils.causal_mask(decoder_input.size(1)).to(device)
        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        # get next value
        next_val = model.map(out)

        """
        # START
        next_val = next_val.reshape(1, -1)
        # Prepare next_vals tensor
        next_vals = torch.empty(1, 1, input_dim).type_as(source).to(device)
        # Assign next_val to the appropriate portion of next_vals
        # No need for .item(), directly use slicing and indexing
        next_vals[0, 0, 0:num_predicted_features] = next_val

        # Handle the case where input_dim is greater than num_predicted_features,
        # filling the remaining with contextual_vars
        if input_dim > num_predicted_features:
            next_vals[0, 0, num_predicted_features:] = contextual_vars[i, :]

        decoder_input = torch.cat(
            [decoder_input, next_vals],
            dim=1,
        )

        # Assuming decoder_output expects to be expanded with the values from next_val
        # and next_val may contain more than one element.

        # Reshape next_val to ensure it is a 1D tensor with num_predicted_features elements
        # This is important if next_val was previously reshaped to [1, num_predicted_features]
        next_val_flat = next_val.reshape(-1)  # Flatten next_val

        # Create a new tensor for concatenation with the same type as source and on the same device
        # Instead of using fill_(), we directly use next_val_flat ensuring it matches the expected dimensions
        # Ensure next_val_flat is unsqueezed if necessary to match decoder_output's dimensions for concatenation
        new_tensor_for_concat = next_val_flat.type_as(source).to(device)

        # Concatenate along the appropriate dimension
        # Assuming decoder_output and new_tensor_for_concat are compatible for concatenation
        decoder_output = torch.cat(
            [decoder_output, new_tensor_for_concat.unsqueeze(0)],
            dim=1,
        )

        # END
        """
        next_val = next_val[0, -1, 0:num_predicted_features]

        next_vals = torch.empty(1, 1, input_dim).type_as(source).to(device)
        next_vals[0, 0, 0:num_predicted_features] = next_val
        if input_dim > num_predicted_features:
            # print(
            #     f"next_vals[0, 0, num_predicted_features:]: {next_vals[0, 0, num_predicted_features:].size()}"
            # )
            # print(f"contextual_vars[i, :]: {contextual_vars[i, :].size()}")
            next_vals[0, 0, num_predicted_features:] = contextual_vars[i, :]

        decoder_input = torch.cat(
            [decoder_input, next_vals],
            dim=1,
        )

        # print(f"decoder_output.size(): {decoder_output.size()}")
        # print(f"next_val.size(): {next_val.size()}")
        decoder_output = torch.cat(
            [
                decoder_output,
                next_val.unsqueeze(0)
                # torch.empty(1).type_as(source).fill_(next_val.item()).to(device),
            ],
            dim=0,
        )
        """
        next_val = next_val[0, -1, 0:num_predicted_features]

        next_vals = torch.empty(1, 1, input_dim).type_as(source).to(device)
        next_vals[0, 0, 0:num_predicted_features] = next_val.item()
        if input_dim > num_predicted_features:
            print(
                f"next_vals[0, 0, num_predicted_features:]: {next_vals[0, 0, num_predicted_features:].size()}"
            )
            print(f"contextual_vars[i, :]: {contextual_vars[i, :].size()}")
            next_vals[0, 0, num_predicted_features:] = contextual_vars[i, :]

        decoder_input = torch.cat(
            [decoder_input, next_vals],
            dim=1,
        )

        decoder_output = torch.cat(
            [
                decoder_output,
                torch.empty(1).type_as(source).fill_(next_val.item()).to(device),
            ],
        )
        """

        # decoder_input = torch.empty(1, 1, input_dim).type_as(source).to(device)
        # for i in range(len(source[0, -1, :])):
        #     decoder_input[:, :, i] = source[0, -1, i]

        # decoder_input = torch.cat(
        #     [
        #         decoder_input,
        #         torch.empty(1, 1, 1).type_as(source).fill_(next_val.item()).to(device),
        #     ],
        #     dim=1,
        # )
        i += 1

    # NOTE: COMMENTING THIS OUT 2/19
    # Get rid of the contextual features that were used as decoder input
    # predicted_vals = decoder_input[:, :, 0]

    # NOTE: COMMENTING THIS OUT 2/19
    # return predicted_vals.squeeze(0)

    return decoder_output


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
            # print(f"Batch {batch_idx}/{len(training_data)}", flush=True)
            encoder_input, decoder_input, future_data = batch
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            future_data = future_data.to(device)
            label = future_data[:, :, 0:num_predicted_features].squeeze(-1).to(device)
            # label = future_data[:, :, 0].squeeze(-1).to(device)

            # print(f"future_data.size(): {future_data.size()}")
            input_dim = len(future_data[0, 0])
            # print(f"input_dim: {input_dim}")
            context = (
                future_data[:, :, num_predicted_features:].squeeze(0)
                if input_dim > num_predicted_features
                else None
            )  # (output length, input_dim - num_predicted_features)

            encoder_mask = None

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                max_len,
                device,
                input_dim=input_dim,
                contextual_vars=context,
                num_predicted_features=num_predicted_features,
            )

            # print(f"model_out.size(): {model_out.size()}")
            # print(
            #     f"target_reading.size(): {label[:num_predicted_features].squeeze(0).size()}"
            # )

            # source_reading = encoder_input[0]

            target_reading = label[:num_predicted_features].squeeze(0)

            # Assuming input_tensor is of shape [24, n]
            if model_out.shape[1] == 1:
                # Adjust target tensor to have the same second dimension
                target_reading = target_reading.unsqueeze(1)

            mse += F.mse_loss(model_out, target_reading)
            mape += utils.mape_loss(model_out, target_reading)
            rmse += utils.rmse_loss(model_out, target_reading)
            mae += utils.mae_loss(model_out, target_reading)
            # print(f"mape single iter: {utils.mape_loss(model_out, target_reading)}")

            # source_readings.append(source_reading)
            # expected.append(target_reading)
            # predicted.append(model_out)

            # Print the source, target and model output
            # if batch_idx % 10 == 0:
            #     print("-" * console_width)
            #     print(f"{f'SOURCE: ':>12}{encoder_input[0]}")
            #     print(f"{f'TARGET: ':>12}{target_reading}")
            #     print(f"{f'PREDICTED: ':>12}{model_out}")
            if batch_idx % 1 == 0:
                source_readings.append(encoder_input[0])
                expected.append(target_reading)
                predicted.append(model_out)

            # if count == num_examples:
            #     print("-" * console_width)
            #     break

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


def get_model(config):
    model = build_transformer(
        len(config["predicted_features"])
        + len(config["continuous_vars"])
        + sum(config["discrete_embedding_dims"]),
        # 1 + len(config["continuous_vars"]) + sum(config["discrete_embedding_dims"]),
        config["discrete_var_dims"],
        config["discrete_embedding_dims"],
        config["enc_seq_len"],
        config["output_sequence_length"],
        num_predicted_features=len(config["predicted_features"]),
        d_model=config["d_model"],
        N=config["N"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        offset=len(config["predicted_features"]) + len(config["continuous_vars"]),
    )
    return model

    # def train_model(config):
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

    print(f"Continuous vars: {config['continuous_vars']}")
    print(f"Discrete vars: {config['discrete_vars']}")

    print(f"Retrieving data from {config['data_path']}")
    df = pd.read_csv(config["data_path"])
    print(f"Data retrieved.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # NOTE: MOVED NORMALIZATION INTO THE DATA PREPROCESSING MODULE
    # Z-score normalization
    # mean_demand = df["Demand (MWh)"].mean()
    # std_demand = df["Demand (MWh)"].std()

    # df["Normalized Demand"] = (df["Demand (MWh)"] - mean_demand) / std_demand
    data = df

    print(data.head())

    # Remove test data from dataset
    training_data = data[: -(round(len(data) * config["test_size"]))]
    # Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
    # Should be training data indices only
    training_indices = utils.get_indices_entire_sequence(
        data=training_data,
        window_size=config["enc_seq_len"] + config["output_sequence_length"],
        step_size=config["step_size"],
    )

    # Define input variables
    # input_variables = [config["target_col_name"]] + config["exogenous_vars"]
    input_variables = (
        [config["target_col_name"]]
        + config["continuous_vars"]
        + config["discrete_vars"]
    )

    discrete_var_dims = utils.calculate_discrete_dims(data, config["discrete_vars"])
    print(f"discrete_var_dims: {discrete_var_dims}")
    config["discrete_var_dims"] = discrete_var_dims
    config["discrete_embedding_dims"] = [
        round(1.6 * math.sqrt(dim)) for dim in discrete_var_dims
    ]

    # Ensure all continuous vars are normalized
    training_data = utils.normalize_continuous_vars(
        data=training_data, var_names=config["continuous_vars"]
    )
    print("Training data normalized.")
    print(f"Input variables: {input_variables}")

    # Making instance of custom dataset class
    training_data = dataset.TransformerDataset(
        data=torch.tensor(training_data[input_variables].values).float(),
        indices=training_indices,
        enc_seq_len=config["enc_seq_len"],
        # dec_seq_len=config["dec_seq_len"],
        target_seq_len=config["output_sequence_length"],
    )
    training_data = DataLoader(training_data, config["batch_size"])
    encoder_mask = None
    # encoder_mask = utils.generate_square_subsequent_mask(
    #     dim1=config["output_sequence_length"], dim2=config["enc_seq_len"]
    # ).to(device)

    decoder_mask = utils.generate_square_subsequent_mask(
        dim1=config["output_sequence_length"], dim2=config["output_sequence_length"]
    ).to(device)

    model = get_model(config).to(device)

    print("Got model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    global_step = 0
    if config["loss_fn"] == "mse":
        loss_fn = nn.MSELoss().to(device)
    elif config["loss_fn"] == "huber":
        loss_fn = nn.HuberLoss().to(device)
    else:
        loss_fn = nn.MSELoss().to(device)

    print("Beginning training...")
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        torch.cuda.empty_cache()
        model.train()

        for batch_idx, batch in enumerate(training_data):
            # print(f"Batch {batch_idx}/{len(training_data)}", flush=True)
            encoder_input, decoder_input, label = batch
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            # label = label.to(device)
            # label size: (batch, output_seq_length, input_dim)
            label = label[:, :, 0].squeeze(-1).to(device)
            # label size: (batch, output_seq_length)
            # print(f"encoder_input.size(): {encoder_input.size()}")
            # print(f"decoder_input.size(): {decoder_input.size()}")
            # print(f"label.size(): {label.size()}")

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (B, seq_len, d_model)
            # print(f"Size of encoder output: {encoder_output.size()}")
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (B, seq_len, d_model)
            # print(f"Size of decoder output: {decoder_output.size()}")
            map_output = model.map(decoder_output)
            # print(f"Size of map output: {map_output.size()}")

            # Flatten tensors and then compute MSE loss
            loss = loss_fn(map_output.view(-1), label.view(-1))
            # batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(training_data)

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_epoch_loss:.4f}")
        # print(f"output ex: {map_output}")

    output_path = config["output_dir"] + config["experiment_name"] + ".pt"
    print(f"Saving model to {output_path}")
    print("...")
    torch.save(model.state_dict(), output_path)
    print("Model successfully saved.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    run_training_and_validation(config)
