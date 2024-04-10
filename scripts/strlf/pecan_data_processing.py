import os
import pandas as pd
import numpy as np
from datetime import datetime


def clean_rows(df):
    df_cleaned = df.dropna(subset=["load"])

    df_cleaned["load"] = df_cleaned["load"].str.replace(",", "")
    df_cleaned["load"] = pd.to_numeric(df_cleaned["load"], errors="coerce")
    df_cleaned["load"] = df_cleaned["load"].astype(int)

    df_cleaned["load"] = df_cleaned["load"].replace(0, pd.NA)
    df_cleaned["load"].fillna(method="ffill", inplace=True)

    mean_val = df_cleaned["load"].mean()
    std_dev = df_cleaned["load"].std()
    print(f"mean_val: {mean_val}")
    print(f"std_dev: {std_dev}")

    # Reset the index to return the 'timestamp' column to the DataFrame
    df_cleaned.reset_index(inplace=True)

    return df_cleaned


def get_final_table():
    """
    Given an arbitrary number of small tables downloaded from eia.gov,
    this concatenates all of them and sorts by time, retuning a final
    concatenated table.
    """

    directory_path = f"./data/pecanstreet/15minute_data_austin/"
    print("Loading data...")
    all_data = pd.read_csv(
        os.path.join(directory_path, "15minute_data_austin.csv"),
        parse_dates=["local_15min"],
    )
    print("Loaded.")

    # read in the metadata file, skip the 2nd row because it has the comments further describing the headers
    metadata = pd.read_csv(os.path.join(directory_path, +"metadata.csv"), skiprows=[1])

    # filter down to our houses of interest. Active, Austin-based, has complete data, and has the grid circuit
    dataids = metadata[
        metadata.active_record.eq("yes")
        & metadata.city.eq("Austin")
        & metadata.egauge_1min_data_availability.isin(
            ["100%", "99%", "98%", "97%", "96%", "95%"]
        )
        & metadata.grid.eq("yes")
    ]

    filt = all_data[all_data.dataid.isin(dataids.dataid)]
    filt["local_15min"] = pd.to_datetime(
        filt["local_15min"], utc=True, infer_datetime_format=True
    )
    filt["local_15min"] = filt["local_15min"].dt.tz_convert("US/Central")

    cols = [
        "dataid",
        "local_15min",
        "air1",
        "furnace1",
        "clotheswasher1",
        "microwave1",
        "refrigerator1",
    ]
    filt2 = filt[cols].fillna(0)
    cols_to_sum = ["air1", "furnace1", "clotheswasher1", "microwave1", "refrigerator1"]
    filt2["total"] = filt2[cols_to_sum].sum(axis=1)
    filt2["year"] = filt2["local_15min"].dt.year
    filt2["month"] = filt2["local_15min"].dt.month
    filt2["day"] = filt2["local_15min"].dt.day
    filt2["hour"] = filt2["local_15min"].dt.hour
    filt2["minute"] = filt2["local_15min"].dt.minute
    filt2["hourminute"] = filt2["local_15min"].dt.strftime("%H:%M")

    # Z-score normalization

    filt2["max_total_per_zone"] = filt2.groupby("dataid")["total"].transform("max")

    # filt2["std_total_per_zone"] = filt2.groupby("dataid")["total"].transform("std")

    # Apply normalization formula for each zone
    filt2["total_normalized"] = (filt2["total"]) / filt2["max_total_per_zone"]

    filt2.drop(["max_total_per_zone"], axis=1, inplace=True)

    return filt2


def create_table():
    df = get_final_table()
    print(df.head())
    df.to_csv(f"./data/final_tables/gefcom12/gefcom12.csv", index=False)
    return df


create_table()
