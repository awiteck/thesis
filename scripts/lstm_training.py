import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_sequences_with_exog(data, exog_data, input_length, output_length):
    xs = []
    ys = []
    for i in range(len(data) - input_length - output_length + 1):
        x_load = data[i : (i + input_length)]
        x_exog = exog_data[i : (i + input_length)]
        y = data[(i + input_length) : (i + input_length + output_length)]

        # Assuming exog_data is a numpy array of shape (n_samples, n_features)
        x_combined = np.hstack(
            (x_load.reshape(-1, 1), x_exog)
        )  # Reshape load to match exog shape and combine

        xs.append(x_combined)
        ys.append(y)
    return np.array(xs), np.array(ys)


def train(df_train, df_test, input_length=24, output_length=24):
    exog_vars = [
        "temperature",
        "cloudcover",
        "humidity",
        "windspeed",
        "temperature_24h_ahead",
        "cloudcover_24h_ahead",
        "humidity_24h_ahead",
        "windspeed_24h_ahead",
    ]
    exog_train = df_train[["feature1", "feature2"]].values  # Example exogenous features
    exog_test = df_test[["feature1", "feature2"]].values

    X_train, y_train = create_sequences_with_exog(
        df_train["load"].values, exog_train, input_length, output_length
    )


X_test, y_test = create_sequences_with_exog(
    df_test["load"].values, exog_test, input_length, output_length
)


X_test, y_test = create_sequences(df_test["load"].values, input_length, output_length)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader instances
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
