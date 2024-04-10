import pandas as pd
import requests
import csv
import os
import numpy as np

API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_weather_data(latitude, longitude, timeStart, timeEnd, timezone):
    response = requests.get(
        API_BASE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": timeStart,
            "end_date": timeEnd,
            "hourly": "temperature_2m,relativehumidity_2m,cloudcover,windspeed_10m,direct_radiation,diffuse_radiation,global_tilted_irradiance",
            "timezone": timezone,
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
        diffuse_radiation = data["hourly"]["diffuse_radiation"]
        global_radiation = data["hourly"]["global_tilted_irradiance"]
        # Create a DataFrame from the lists
        weather_data = pd.DataFrame(
            {
                "Time": times,
                "temperature": temperatures,
                "humidity": humidity,
                "cloudcover": cloudcover,
                "windspeed": windspeed_10m,
                "direct_radiation": direct_radiation,
                "diffuse_radiation": diffuse_radiation,
                "global_radiation": global_radiation,
            }
        )
        weather_data["Time"] = pd.to_datetime(weather_data["Time"])
        return weather_data
    else:
        print(f"Error {response.status_code}: {response.text}")


def create_table(region):
    assert region in ["ERCOT", "NYISO"], "region must be ERCOT or NYISO"
    if region == "ERCOT":
        wind = pd.read_csv(
            "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/ERCOT/Wind/Actual/wind_actual_1h_site_2017_2018_utc.csv"
        )
        forecast = pd.read_csv(
            "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/ERCOT/Wind/Day-ahead/wind_day_ahead_forecast_site_2018_utc.csv"
        )
        metadata = pd.read_excel(
            "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/ERCOT/MetaData/wind_meta.xlsx"
        )
        # metadata = metadata[metadata["proposed"] == "existing"]
        time_offset = pd.Timedelta(hours=-6)
        timezone = "America/Chicago"
    else:
        wind = pd.read_csv(
            "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/NYISO/Wind/Actual/wind_actual_1h_site_2018_2019_utc.csv"
        )
        forecast = pd.read_csv(
            "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/NYISO/Wind/Day-ahead/wind_day_ahead_forecast_site_2018_2019_utc.csv"
        )
        metadata = pd.read_csv(
            "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/NYISO/MetaData/wind_meta.csv"
        )
        time_offset = pd.Timedelta(hours=-5)
        timezone = "America/New_York"
    # Transform the DataFrame
    wind_long = pd.melt(
        wind, id_vars=["Time"], var_name="source", value_name="generation"
    )

    forecast = forecast.drop(columns=["Issue_time"])

    forecast_long = pd.melt(
        forecast, id_vars=["Forecast_time"], var_name="source", value_name="forecast"
    )

    # Rename the column containing time information appropriately if needed
    wind_long.rename(columns={"Time": "time"}, inplace=True)
    wind = wind_long

    # Rename the column containing time information appropriately if needed
    forecast_long.rename(columns={"Forecast_time": "time"}, inplace=True)
    forecast = forecast_long

    wind = pd.merge(
        wind,
        forecast,
        left_on=["time", "source"],
        right_on=["time", "source"],
        how="left",
    )

    # Merge on the source/site_ids to add the metadata to the solar data
    wind = pd.merge(
        wind, metadata, left_on="source", right_on="Facility.Name", how="left"
    )
    print("Merged...")
    wind["normalized_generation"] = wind["generation"] / wind["Capacity"]
    wind["normalized_forecast"] = wind["forecast"] / wind["Capacity"]
    print("Normalized...")
    wind["time"] = pd.to_datetime(wind["time"], utc=False)
    print("Time converted...")
    # Apply the function to adjust times

    # Apply the offset to the entire 'time' column directly
    wind["adjusted_time"] = wind["time"] + time_offset
    print("Time adjusted...")

    print("Getting weather data...")

    time_start = wind["adjusted_time"].min().strftime("%Y-%m-%d")
    time_end = wind["adjusted_time"].max().strftime("%Y-%m-%d")

    print(wind.head())

    # Placeholder for storing weather data DataFrames
    weather_dfs = []

    for index, row in metadata.iterrows():
        print(f"{index}: Checking data for {row['Facility.Name']}")
        file_path = (
            f"./data/final_tables/{region}_wind_weather/{row['Facility.Name']}.csv"
        )

        # Check if the weather data file for the current site already exists
        if os.path.exists(file_path):
            # If it exists, read the weather data from the file
            print(f"Loading existing weather data for {row['Facility.Name']}")
            weather_data = pd.read_csv(file_path)
        else:
            # If it does not exist, retrieve weather data via API call
            print(f"Retrieving data for {row['Facility.Name']} via API")
            weather_data = get_weather_data(
                row["lati"],
                row["longi"],
                time_start,
                time_end,
                timezone=timezone,
            )

            # Add a column to identify the source
            weather_data["source"] = row["Facility.Name"]

            # Save the retrieved weather data to a file
            os.makedirs(
                os.path.dirname(file_path), exist_ok=True
            )  # Ensure the directory exists
            weather_data.to_csv(file_path, index=False)
        # Append the DataFrame to the list
        weather_dfs.append(weather_data)

    # Concatenate all weather DataFrames into a single DataFrame
    all_weather_data = pd.concat(weather_dfs, ignore_index=True)
    all_weather_data["direct_radiation"] = all_weather_data["direct_radiation"] / max(
        all_weather_data["direct_radiation"]
    )
    all_weather_data["diffuse_radiation"] = all_weather_data["diffuse_radiation"] / max(
        all_weather_data["diffuse_radiation"]
    )
    all_weather_data["global_radiation"] = all_weather_data["global_radiation"] / max(
        all_weather_data["global_radiation"]
    )

    wind["adjusted_time"] = pd.to_datetime(wind["adjusted_time"]).dt.tz_localize(None)
    # Convert 'Time' in all_weather_data to datetime and adjust to match ercot_solar's 'adjusted_time'
    all_weather_data["Time"] = pd.to_datetime(all_weather_data["Time"]).dt.tz_localize(
        None
    )
    # Assuming all_weather_data['Time'] is in the same timezone as ercot_solar['adjusted_time']

    print("Merging wind with weather...")
    # Merge the weather data with ercot_solar
    wind_with_weather = pd.merge(
        wind,
        all_weather_data,
        left_on=["adjusted_time", "source"],
        right_on=["Time", "source"],
        how="left",
    )
    print("Merge complete. Dropping columns and saving...")
    wind_with_weather["time"] = wind_with_weather["adjusted_time"]

    columns_to_drop = [
        "adjusted_time",
        "Facility.Name",
        "Capacity",
        "lati",
        "longi",
        "Region",
        "Time",
    ]
    wind_with_weather = wind_with_weather.drop(columns=columns_to_drop)
    wind_with_weather["year"] = wind_with_weather["time"].dt.year
    wind_with_weather["month"] = wind_with_weather["time"].dt.month
    wind_with_weather["day_of_week"] = wind_with_weather["time"].dt.dayofweek
    wind_with_weather["day"] = wind_with_weather["time"].dt.day
    wind_with_weather["hour"] = wind_with_weather["time"].dt.hour
    final_df = wind_with_weather[wind_with_weather["year"] > 2016]

    final_df.sort_values(by=["source", "time"], inplace=True)
    final_df["direct_radiation_24h_ahead"] = final_df.groupby("source")[
        "direct_radiation"
    ].shift(-24)
    final_df["diffuse_radiation_24h_ahead"] = final_df.groupby("source")[
        "diffuse_radiation"
    ].shift(-24)
    final_df["global_radiation_24h_ahead"] = final_df.groupby("source")[
        "global_radiation"
    ].shift(-24)
    final_df["windspeed_24h_ahead"] = final_df.groupby("source")["windspeed"].shift(-24)
    final_df["temperature_24h_ahead"] = final_df.groupby("source")["temperature"].shift(
        -24
    )

    final_df = final_df.dropna(
        subset=[
            "direct_radiation_24h_ahead",
            "diffuse_radiation_24h_ahead",
            "global_radiation_24h_ahead",
            "windspeed_24h_ahead",
            "temperature_24h_ahead",
        ]
    )

    # List of columns to add noise to
    columns_of_interest = [
        "direct_radiation_24h_ahead",
        "diffuse_radiation_24h_ahead",
        "global_radiation_24h_ahead",
        "windspeed_24h_ahead",
        "temperature_24h_ahead",
    ]

    # Adding 5% noise
    for column in columns_of_interest:
        noise = np.random.normal(0, 0.05, size=len(final_df[column])) * final_df[column]
        final_df[column + "_5p"] = np.maximum(0, final_df[column] + noise)

    # Adding 10% noise
    for column in columns_of_interest:
        noise = (
            np.random.normal(0, 0.10, size=final_df[column].shape) * final_df[column]
        )
        final_df[column + "_10p"] = np.maximum(0, final_df[column] + noise)

    # Adding 20% noise
    for column in columns_of_interest:
        noise = (
            np.random.normal(0, 0.20, size=final_df[column].shape) * final_df[column]
        )
        final_df[column + "_20p"] = np.maximum(0, final_df[column] + noise)

    # Adding 50% noise
    for column in columns_of_interest:
        noise = (
            np.random.normal(0, 0.50, size=final_df[column].shape) * final_df[column]
        )
        final_df[column + "_50p"] = np.maximum(0, final_df[column] + noise)

    # Adding 100% noise
    for column in columns_of_interest:
        noise = (
            np.random.normal(0, 1.00, size=final_df[column].shape) * final_df[column]
        )
        final_df[column + "_100p"] = np.maximum(0, final_df[column] + noise)

    final_df_train = final_df[(final_df["year"] < 2018) | (final_df["month"] < 12)]
    final_df_test = final_df[(final_df["year"] == 2018) & (final_df["month"] == 12)]
    final_df_train.to_csv(
        f"./data/final_tables/{region}_wind/{region}_wind_train.csv",
        index=False,
    )
    final_df_test.to_csv(
        f"./data/final_tables/{region}_wind/{region}_wind_test.csv",
        index=False,
    )
    print("Saved.")
    return final_df


create_table("ERCOT")
