import os
import pandas as pd
import numpy as np
from datetime import datetime
import requests

API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


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


def get_weather_data(df, filename, time_start, time_end):
    # Split the filename by underscore
    parts = filename.split("_")

    # data_type = parts[0]
    # if data_type != "Actual":

    # Assuming the coordinates are always in the second and third positions after splitting
    latitude = float(parts[1])
    longitude = float(parts[2])

    response = requests.get(
        API_BASE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": time_start,
            "end_date": time_end,
            "hourly": "temperature_2m,relativehumidity_2m,cloudcover,windspeed_10m,direct_radiation",
        },
    )
    if response.status_code == 200:
        data = response.json()
        times = data["hourly"]["time"]
        temperatures = data["hourly"]["temperature_2m"]
        humidity = data["hourly"]["relativehumidity_2m"]
        cloudcover = data["hourly"]["cloudcover"]
        windspeed_10m = data["hourly"]["windspeed_10m"]
        direct_radiation = data["hourly"]["direct_radiation"]
        # Create a DataFrame from the lists
        weather_data = pd.DataFrame(
            {
                "Time": times,
                "temperature": temperatures,
                "humidity": humidity,
                "cloudcover": cloudcover,
                "windspeed": windspeed_10m,
                "direct_radiation": direct_radiation,
            }
        )
        weather_data["Time"] = pd.to_datetime(weather_data["Time"])
        # Merge based on timestamp
        # Create auxiliary columns for merging, truncating to the hour
        df["MergeKey"] = df["LocalTime"].dt.floor("H")
        weather_data["MergeKey"] = weather_data["Time"].dt.floor("H")
        print(df.head())
        print(weather_data.head())

        # Merge using the auxiliary 'MergeKey' columns
        df_merged = pd.merge(
            df, weather_data, left_on="MergeKey", right_on="MergeKey", how="left"
        )

        # Optionally, drop the 'MergeKey' column if it's no longer needed
        df_merged.drop("MergeKey", axis=1, inplace=True)
        df_merged.drop(columns=["Time"], inplace=True)

        return df_merged
    else:
        print(f"Error {response.status_code}: {response.text}")


def get_final_table():
    directory_path = f"./data/solar/tx-east-pv-2006/"
    all_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]
    # Initialize an empty list to store the dataframes
    dfs = []

    # Read each CSV file and append it to the list
    # NOTE: JUST DO FIRST FEW SOLAR PLANTS FOR NOW.
    for filename in all_files[:30]:
        parts = filename.split("_")

        data_type = parts[0]
        if data_type != "Actual":
            continue
        df = pd.read_csv(os.path.join(directory_path, filename))
        df["LocalTime"] = pd.to_datetime(df["LocalTime"], format="%m/%d/%y %H:%M")

        plant_id = parts[1] + "_" + parts[2] + "_" + parts[5]
        df["plant_id"] = plant_id
        df["year"] = df["LocalTime"].dt.year
        df["month"] = df["LocalTime"].dt.month
        df["day"] = df["LocalTime"].dt.day
        df["hour"] = df["LocalTime"].dt.hour
        df["minute"] = df["LocalTime"].dt.minute
        time_start = df["LocalTime"].min().strftime("%Y-%m-%d")
        time_end = df["LocalTime"].max().strftime("%Y-%m-%d")
        df_with_weather = get_weather_data(df, filename, time_start, time_end)
        print(df_with_weather.head())
        dfs.append(df_with_weather)

    # Concatenate all the dataframes together
    final_df = pd.concat(dfs, ignore_index=True)
    print("Concatenated...")
    final_df["max_power_per_plant"] = final_df.groupby("plant_id")[
        "Power(MW)"
    ].transform("max")
    final_df["power_normalized"] = (final_df["Power(MW)"]) / final_df[
        "max_power_per_plant"
    ]
    final_df.drop(columns=["max_power_per_plant"], inplace=True)

    print(final_df.head())

    return final_df


def create_table():
    df = get_final_table()
    print(df.head())
    df.to_csv(f"./data/final_tables/solar/tx-east-pv-2006.csv", index=False)
    return df


create_table()
