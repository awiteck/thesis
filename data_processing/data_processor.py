import demand_helper as demand
import weather_helper as weather
import numpy as np
import pandas as pd


def create_table(region_name, latitude, longitude):
    # Get all available electric load data for the region first
    load_data = demand.get_concatenated_table(region_name)

    print(load_data.head())

    time_start = load_data["timestamp"].min().strftime("%Y-%m-%d")
    time_end = load_data["timestamp"].max().strftime("%Y-%m-%d")

    weather_data = weather.getWeatherData(latitude, longitude, time_start, time_end)

    print(weather_data.head())
    # Convert timestamp columns to datetime objects
    load_data["timestamp"] = pd.to_datetime(load_data["timestamp"])
    weather_data["Time"] = pd.to_datetime(weather_data["Time"])

    # Merge based on timestamp
    df_merged = pd.merge(
        load_data, weather_data, left_on="timestamp", right_on="Time", how="left"
    )

    # Drop the redundant 'Time' column (optional)
    df_merged.drop(columns=["Time"], inplace=True)
    df_merged["day_of_week"] = df_merged["timestamp"].dt.dayofweek
    df_merged["hour"] = df_merged["timestamp"].dt.hour
    df_merged["month"] = df_merged["timestamp"].dt.month
    df_merged.to_csv(
        f"./data/final_tables/{region_name}/{region_name}.csv", index=False
    )
    return df_merged


df = create_table("banc", 38.5816, -121.489906)
print(df.head())

df = create_table("isne", 43.1026, -71.6165)
print(df.head())

df = create_table("erco", 30.5434, -97.1943)
print(df.head())
