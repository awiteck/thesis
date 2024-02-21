import torch
import torch.nn as nn
from config import get_config
from train_helper import get_model, run_validation
import pandas as pd
import utils
import dataset
from torch.utils.data import DataLoader
import math

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
config = get_config()

# Load dataset
print(f"Retrieving data from {config['data_path']}")
df = pd.read_csv(config["data_path"])
print(f"Data retrieved.")
df["timestamp"] = pd.to_datetime(df["timestamp"])

data = df

discrete_var_dims = utils.calculate_discrete_dims(data, config["discrete_vars"])
print(f"discrete_var_dims: {discrete_var_dims}")
config["discrete_var_dims"] = discrete_var_dims
config["discrete_embedding_dims"] = [
    round(1.6 * math.sqrt(dim)) for dim in discrete_var_dims
]

model = get_model(config).to(device)

# Load the pretrained weights
print("Loading model")
model_filename = config["output_dir"] + config["experiment_name"] + ".pt"
model.load_state_dict(torch.load(model_filename))
print("Model loaded")

# Remove test data from dataset
validation_data = data[-(round(len(data) * config["test_size"])) :]
# Make list of (start_idx, end_idx) pairs that are used to slice the time series sequence into chunkc.
# Should be training data indices only
validation_indices = utils.get_indices_entire_sequence(
    data=validation_data,
    window_size=config["enc_seq_len"] + config["output_sequence_length"],
    step_size=config["step_size"],
)

# Define input variables
input_variables = (
    [config["target_col_name"]] + config["continuous_vars"] + config["discrete_vars"]
)

discrete_var_dims = utils.calculate_discrete_dims(data, config["discrete_vars"])
config["discrete_var_dims"] = discrete_var_dims


# Ensure all continuous vars are normalized
validation_data = utils.normalize_continuous_vars(
    data=validation_data, var_names=config["continuous_vars"]
)

# Making instance of custom dataset class
validation_data = dataset.TransformerDataset(
    data=torch.tensor(validation_data[input_variables].values).float(),
    indices=validation_indices,
    enc_seq_len=config["enc_seq_len"],
    # dec_seq_len=config["dec_seq_len"],
    target_seq_len=config["output_sequence_length"],
)
validation_data = DataLoader(validation_data, 1)

run_validation(
    model,
    validation_data,
    config["output_sequence_length"],
    device,
    num_examples=10,
)
