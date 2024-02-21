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


def get_holidays(df):
    directory_path = f"./data/GEFCOM2012/Load/"
    holiday_df = pd.read_csv(os.path.join(directory_path, "Holiday_list_new.csv"))
    # Initialize an empty list to store holiday dates
    holiday_dates = []
    # Process each column (year) in the holiday DataFrame
    for column in holiday_df.columns:
        try:
            year = int(column)  # Convert column name to integer year
            for value in holiday_df[column].dropna():  # Drop NaN values
                # Parse the date string into a datetime object
                date_str = f"{value}, {year}"
                # Assuming the format in the CSV is consistent as "DayOfWeek, MonthName Day"
                date_obj = datetime.strptime(date_str, "%A, %B %d, %Y")
                holiday_dates.append(date_obj)
        except:
            continue
    print(holiday_dates)
    # Now, prepare df_cleaned by creating a datetime column from "year", "month", "day"
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    # Step 4: Cross-Check and Assign "is_holiday"
    # Initialize the column with 0s
    df["is_holiday"] = 0

    # Set 'is_holiday' to 1 if the date matches any holiday date
    for holiday in holiday_dates:
        df.loc[df["date"] == holiday, "is_holiday"] = 1

    return df


def get_final_table():
    """
    Given an arbitrary number of small tables downloaded from eia.gov,
    this concatenates all of them and sorts by time, retuning a final
    concatenated table.
    """

    directory_path = f"./data/GEFCOM2012/Load/"

    df = pd.read_csv(os.path.join(directory_path, "Load_history.csv"))
    df_temps = pd.read_csv(os.path.join(directory_path, "temperature_history.csv"))

    # Melt the DataFrame to transform hour columns into rows
    loads = df.melt(
        id_vars=["zone_id", "year", "month", "day"], var_name="hour", value_name="load"
    )

    # Extract hour number from the 'hour' column and convert to integer
    loads["hour"] = loads["hour"].str.extract("(\d+)").astype(int)

    # Sort and optionally reset index
    loads = loads.sort_values(
        by=["zone_id", "year", "month", "day", "hour"]
    ).reset_index(drop=True)

    final_df = get_holidays(clean_rows(loads))

    # Z-score normalization

    final_df["mean_demand_per_zone"] = final_df.groupby("zone_id")["load"].transform(
        "mean"
    )
    final_df["std_demand_per_zone"] = final_df.groupby("zone_id")["load"].transform(
        "std"
    )
    # Apply normalization formula for each zone
    final_df["load_normalized"] = (
        final_df["load"] - final_df["mean_demand_per_zone"]
    ) / final_df["std_demand_per_zone"]

    final_df.drop(["mean_demand_per_zone", "std_demand_per_zone"], axis=1, inplace=True)

    # mean_demand = final_df["load"].mean()
    # std_demand = final_df["load"].std()

    # final_df["load_normalized"] = (final_df["load"] - mean_demand) / std_demand

    return final_df


def create_table():
    df = get_final_table()
    print(df.head())
    df.to_csv(f"./data/final_tables/gefcom12/gefcom12.csv", index=False)
    return df


create_table()
