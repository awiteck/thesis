import os
import pandas as pd
import numpy as np


def clean_rows(df):
    # Replace 0 values with NaN
    df["Demand (MWh)"] = df["Demand (MWh)"].replace(0, pd.NA)

    # Replace negative values in 'Demand (MWh)' with NaN
    df.loc[df["Demand (MWh)"] < 0, "Demand (MWh)"] = pd.NA

    mean_val = df["Demand (MWh)"].mean()
    std_dev = df["Demand (MWh)"].std()
    print(f"mean_val: {mean_val}")
    print(f"std_dev: {std_dev}")

    threshold = mean_val + 5 * std_dev
    # Replace extreme outliers with NaN
    df.loc[df["Demand (MWh)"] > threshold, "Demand (MWh)"] = pd.NA

    # Ensure the 'timestamp' column is of datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Generate the full date-time range with an hourly frequency
    full_range = pd.date_range(
        start="2015-07-01 01:00:00", end="2023-01-01 00:00:00", freq="H"
    )

    # Set the DataFrame's index to the 'timestamp' column
    df.set_index("timestamp", inplace=True)

    # Reindex the DataFrame with the full date-time range
    df = df.reindex(full_range)

    # Forward fill missing values for each column
    columns_to_fill = ["Demand (MWh)", "Demand Forecast (MWh)", "Net Generation (MWh)"]
    for col in columns_to_fill:
        df[col].fillna(method="ffill", inplace=True)

    # Reset the index to return the 'timestamp' column to the DataFrame
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)

    return df


def get_concatenated_table(region_name):
    """
    Given an arbitrary number of small tables downloaded from eia.gov,
    this concatenates all of them and sorts by time, retuning a final
    concatenated table.
    """

    directory_path = f"./data/base_tables/{region_name}/"
    all_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

    # Initialize an empty list to store the dataframes
    dfs = []

    # Read each CSV file and append it to the list
    for filename in all_files:
        df = pd.read_csv(os.path.join(directory_path, filename))
        dfs.append(df)

    # Concatenate all the dataframes together
    final_df = pd.concat(dfs, ignore_index=True)

    # Convert each time to a pd timestamp object. Then we can sort based on this.
    # NOTE: we disregard any difference between PDT and PST here.
    final_df["Timestamp (Hour Ending)"] = (
        final_df["Timestamp (Hour Ending)"]
        .str.replace("a.m.", "AM")
        .str.replace("p.m.", "PM")
        .str.replace(" PDT", "")
        .str.replace(" PST", "")
        .str.replace(" EDT", "")
        .str.replace(" EST", "")
        .str.replace(" CDT", "")
        .str.replace(" CST", "")
    )

    # Convert the modified 'Timestamp (Hour Ending)' column to datetime format
    final_df["Timestamp (Hour Ending)"] = pd.to_datetime(
        final_df["Timestamp (Hour Ending)"], format="%m/%d/%Y %I %p", errors="coerce"
    )

    # Drop unwanted columns
    columns_to_drop = [
        "Percent Change from Prior Hour",
        "Prior Hour Demand (MWh)",
        "Region Code",
        "Region Type",
        "Selected Hour Demand (MWh)",
        "Selected Hour Timestamp (Hour Ending)",
        "Total Interchange (MWh)",
        "BA Code",
    ]
    print("DROPPING...")
    # Filter the list to include only columns that exist in the DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in final_df.columns]

    final_df.drop(columns=columns_to_drop, inplace=True)

    print(final_df.head)
    # Rename columns
    final_df.rename(columns={"Timestamp (Hour Ending)": "timestamp"}, inplace=True)

    final_df = final_df.sort_values(by="timestamp")
    final_df = final_df.drop_duplicates(subset=["timestamp"])
    final_df = clean_rows(final_df)
    final_df = clean_rows(final_df)
    final_df["Region"] = region_name

    # Z-score normalization
    mean_demand = final_df["Demand (MWh)"].mean()
    std_demand = final_df["Demand (MWh)"].std()

    print(f"max: {max(final_df['Demand (MWh)'])}")
    print(f"threshold: {mean_demand+3*std_demand}")

    final_df["Normalized Demand"] = (
        final_df["Demand (MWh)"] - mean_demand
    ) / std_demand

    return final_df
