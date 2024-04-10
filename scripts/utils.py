import torch
import pandas as pd


def mape_loss(output, target):
    """
    Calculate Mean Absolute Percentage Error (MAPE) loss
    NOTE: Loss is not multiplied by 100

    Args:
    output (torch.Tensor): Predicted values from the model.
    target (torch.Tensor): True values.

    Returns:
    torch.Tensor: MAPE loss
    """
    # Avoid division by zero
    epsilon = 1e-7
    # Calculate the absolute percentage error
    percentage_error = torch.abs((target - output) / (target + epsilon))
    # Calculate the mean of these percentage errors
    loss = torch.mean(percentage_error)
    return loss


def rmse_loss(output, target):
    """
    Calculate Root Mean Square Error (RMSE) loss

    Args:
    output (torch.Tensor): Predicted values from the model.
    target (torch.Tensor): True values.

    Returns:
    torch.Tensor: RMSE loss
    """
    loss = torch.sqrt(torch.mean((output - target) ** 2))
    return loss


def mae_loss(output, target):
    """
    Calculate Mean Absolute Error (MAE) loss

    Args:
    output (torch.Tensor): Predicted values from the model.
    target (torch.Tensor): True values.

    Returns:
    torch.Tensor: MAE loss
    """
    loss = torch.mean(torch.abs(output - target))
    return loss


def remove_overlapping_intervals(primary_list, secondary_list):
    def is_overlapping(interval1, interval2):
        # Returns True if interval1 overlaps with interval2
        return interval1[0] < interval2[1] and interval1[1] > interval2[0]

    result = (
        []
    )  # List to store intervals from secondary_list that do not overlap with primary_list
    for secondary_interval in secondary_list:
        overlap = False
        for primary_interval in primary_list:
            if is_overlapping(primary_interval, secondary_interval):
                # print(f"overlap between {primary_interval} and {secondary_interval}")
                overlap = True
                break
        if not overlap:
            result.append(secondary_interval)
    return result


# def remove_overlapping_indices(primary_set, secondary_set, T):
#     """
#     Remove indices from the secondary_set that overlap with any index in the primary_set,
#     considering the input length T.
#     """
#     to_remove = set()
#     for primary_index in primary_set:
#         # Calculate the range of indices that would overlap with primary_index considering T
#         overlapping_range = set(range(primary_index[0] - T + 1, primary_index[1] + T))
#         to_remove |= (
#             overlapping_range & secondary_set
#         )  # Find and add overlapping indices
#     secondary_set -= to_remove  # Remove overlapping indices
#     return secondary_set


def generate_square_subsequent_mask(dim1: int, dim2: int):
    return torch.triu(torch.ones(dim1, dim2) * float("-inf"), diagonal=1)


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def get_indices_entire_sequence(data, window_size: int, step_size: int) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences.

    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences.

    Args:
        num_obs (int): Number of observations (time steps) in the entire
                       dataset for which indices must be generated, e.g.
                       len(data)

        window_size (int): The desired length of each sub-sequence. Should be
                           (input_sequence_length + target_sequence_length)
                           E.g. if you want the model to consider the past 100
                           time steps in order to predict the future 50
                           time steps, window_size = 100+50 = 150

        step_size (int): Size of each step as the data sequence is traversed
                         by the moving window.
                         If 1, the first sub-sequence will be [0:window_size],
                         and the next will be [1:window_size].

    Return:
        indices: a list of tuples
    """

    stop_position = len(data) - 1  # 1- because of 0 indexing

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0
    subseq_last_idx = window_size
    indices = []
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))
        subseq_first_idx += step_size
        subseq_last_idx += step_size
    return indices


def normalize_continuous_vars(data, var_names):
    # mean/std normalize.
    for var in var_names:
        assert var in data.columns, f"Error: {var} is not a column of input dataframe."
        # Z-score normalization
        mean = data[var].mean()
        std = data[var].std()
        print(f"{var}: mean: {mean}, std: {std}")
        data.loc[:, var] = (data[var] - mean) / std
    return data


def normalize_discrete_vars(data, var_names):
    mappings = {}

    for var in var_names:
        assert var in data.columns, f"Error: {var} is not a column of input dataframe."

        print(f"Normalizing {var}")
        # Use factorize to assign a unique index to each category in 'col1'
        codes, uniques = pd.factorize(data[var])

        # Store the mapping from unique values to codes in a dictionary
        mapping = {val: code for code, val in enumerate(uniques)}
        mappings[var] = mapping

        # The 'codes' array contains the encoded values
        data[var] = codes

    for variable, mapping in mappings.items():
        print(f"{variable}: {mapping}")

    return data


def calculate_discrete_dims(data, discrete_col_names):
    res = []
    for col in discrete_col_names:
        assert col in data.columns, f"Error: {col} is not a column of input dataframe."
        res.append(data[col].nunique())
    return res


def remove_overlapping_indices(primary_set, secondary_set, T):
    """
    Remove indices from the secondary_set that overlap with any index in the primary_set,
    considering the input length T.
    """
    to_remove = set()
    for primary_index in primary_set:
        # Calculate the range of indices that would overlap with primary_index considering T
        overlapping_range = set(range(primary_index[0] - T + 1, primary_index[1] + T))
        to_remove |= (
            overlapping_range & secondary_set
        )  # Find and add overlapping indices
    secondary_set -= to_remove  # Remove overlapping indices
    return secondary_set
