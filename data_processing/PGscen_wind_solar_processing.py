import pandas as pd
import requests
import csv
import os
import numpy as np

API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


# This function adjusts the time by the timezone offset
def adjust_timezone(row):
    # Convert timezone from hours to a pandas timedelta
    offset = pd.Timedelta(hours=row["timezone"])
    # Adjust the time
    return row["time"] + offset


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
    ercot_load = pd.read_csv(
        "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/ERCOT/Load/Actual/load_actual_1h_zone_2017_2018_utc.csv"
    )
    ercot_wind = pd.read_csv(
        "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/ERCOT/Wind/Actual/wind_actual_1h_site_2017_2018_utc.csv"
    )
    ercot_solar = pd.read_csv(
        "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/ERCOT/Solar/Actual/solar_actual_1h_site_2017_2018_utc.csv"
    )
    solar_metadata = pd.read_excel(
        "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/ERCOT/MetaData/solar_meta.xlsx"
    )
    wind_metadata = pd.read_excel(
        "/Users/aidanwiteck/Desktop/Princeton/Year 4/Thesis/thesis2/data/PGscen/ERCOT/MetaData/wind_meta.xlsx"
    )

    solar_metadata_region = solar_metadata[solar_metadata["Zone"] == region]
    # (solar_metadata["proposed"] == "Existing") & (solar_metadata["Zone"] == region)
    # ]
    wind_metadata_region = wind_metadata[wind_metadata["Region"] == region]
    # (wind_metadata["Group"] == "Existing")
    #     & (wind_metadata["Region"] == region)
    # ]

    load_region = ercot_load[["Time", region.replace(" ", "_")]]
    solar_region = ercot_solar[["Time"] + solar_metadata_region["site_ids"].tolist()]
    wind_region = ercot_wind[["Time"] + wind_metadata_region["Facility.Name"].tolist()]

    solar_region["total_solar"] = solar_region.drop("Time", axis=1).sum(axis=1)
    wind_region["total_wind"] = wind_region.drop("Time", axis=1).sum(axis=1)

    merged_df = pd.merge(
        wind_region[["Time", "total_wind"]],
        solar_region[["Time", "total_solar"]],
        on="Time",
    )

    # Then, merge the load DataFrame with the previously merged DataFrame (merged_df)
    final_merged_df = pd.merge(
        merged_df, load_region[["Time", region.replace(" ", "_")]], on="Time"
    )
    # First, ensure consistent column naming if necessary
    final_merged_df.rename(
        columns={
            "Time": "time",
            region.replace(" ", "_"): "load",
            "total_wind": "wind",
            "total_solar": "solar",
        },
        inplace=True,
    )

    df = final_merged_df

    time_offset = pd.Timedelta(hours=-6)
    timezone = "America/Chicago"

    print(f"{region}- load mean: {np.mean(df['load'])}")
    print(f"{region}- load std: {np.std(df['load'])}")
    print(f"{region}- wind max: {max(df['wind'])}")
    print(f"{region}- solar max: {max(df['solar'])}")

    # Normalize
    # df["load"] = (df["load"] - min(df["load"])) / (max(df["load"]) - min(df["load"]))
    df["load"] = (df["load"] - np.mean(df["load"])) / np.std(df["load"])
    df["wind"] = df["wind"] / max(df["wind"])
    df["solar"] = df["solar"] / max(df["solar"])

    df["time"] = pd.to_datetime(df["time"], utc=False)
    print("Time converted...")

    # Apply the offset to the entire 'time' column directly
    df["adjusted_time"] = df["time"] + time_offset
    print("Time adjusted...")

    print("Getting weather data...")

    time_start = df["adjusted_time"].min().strftime("%Y-%m-%d")
    time_end = df["adjusted_time"].max().strftime("%Y-%m-%d")

    print(df.head())

    latitude = (
        solar_metadata_region["latitude"].mean() + wind_metadata_region["lati"].mean()
    ) / 2
    longitude = (
        solar_metadata_region["longitude"].mean() + wind_metadata_region["longi"].mean()
    ) / 2
    weather_data = get_weather_data(
        latitude,
        longitude,
        time_start,
        time_end,
        timezone=timezone,
    )

    weather_data["direct_radiation"] = weather_data["direct_radiation"] / max(
        weather_data["direct_radiation"]
    )
    weather_data["diffuse_radiation"] = weather_data["diffuse_radiation"] / max(
        weather_data["diffuse_radiation"]
    )
    weather_data["global_radiation"] = weather_data["global_radiation"] / max(
        weather_data["global_radiation"]
    )

    df["adjusted_time"] = pd.to_datetime(df["adjusted_time"]).dt.tz_localize(None)
    # Convert 'Time' in all_weather_data to datetime and adjust to match ercot_solar's 'adjusted_time'

    weather_data["Time"] = pd.to_datetime(weather_data["Time"]).dt.tz_localize(None)
    # Assuming all_weather_data['Time'] is in the same timezone as ercot_solar['adjusted_time']

    print("Merging with weather...")
    # Merge the weather data with ercot_solar
    df_with_weather = pd.merge(
        df,
        weather_data,
        left_on=["adjusted_time"],
        right_on=["Time"],
        how="left",
    )

    print("Merge complete. Dropping columns and saving...")
    print(df_with_weather.head())
    df_with_weather["time"] = df_with_weather["adjusted_time"]

    columns_to_drop = [
        "adjusted_time",
        "site_ids",
        "AC_capacity_MW",
        "latitude",
        "longitude",
        "elevation",
        "timezone",
        "proposed",
        "Zone",
        "ISO",
        "Time",
    ]
    columns_to_drop = [c for c in columns_to_drop if c in df_with_weather]
    df_with_weather = df_with_weather.drop(columns=columns_to_drop)
    df_with_weather["year"] = df_with_weather["time"].dt.year
    df_with_weather["month"] = df_with_weather["time"].dt.month
    df_with_weather["day_of_week"] = df_with_weather["time"].dt.dayofweek
    df_with_weather["day"] = df_with_weather["time"].dt.day
    df_with_weather["hour"] = df_with_weather["time"].dt.hour
    final_df = df_with_weather[df_with_weather["year"] > 2016]

    final_df.sort_values(by=["time"], inplace=True)
    final_df["direct_radiation_24h_ahead"] = final_df["direct_radiation"].shift(-24)
    final_df["diffuse_radiation_24h_ahead"] = final_df["diffuse_radiation"].shift(-24)
    final_df["global_radiation_24h_ahead"] = final_df["global_radiation"].shift(-24)
    final_df["windspeed_24h_ahead"] = final_df["windspeed"].shift(-24)

    final_df = final_df.dropna(
        subset=[
            "direct_radiation_24h_ahead",
            "diffuse_radiation_24h_ahead",
            "global_radiation_24h_ahead",
            "windspeed_24h_ahead",
        ]
    )
    final_df["region"] = region.replace(" ", "_")

    # List of columns to add noise to
    columns_of_interest = [
        "direct_radiation_24h_ahead",
        "diffuse_radiation_24h_ahead",
        "global_radiation_24h_ahead",
        "windspeed_24h_ahead",
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
        f"./data/final_tables/ercot_region/ercot_{region.replace(' ', '_')}_train.csv",
        index=False,
    )
    final_df_test.to_csv(
        f"./data/final_tables/ercot_region/ercot_{region.replace(' ', '_')}_test.csv",
        index=False,
    )
    print("Saved.")
    return final_df


create_table("Coast")
create_table("East")
create_table("North")
create_table("North Central")
create_table("South")
create_table("South Central")
create_table("West")
create_table("Far West")


directory_path = f"./data/final_tables/ercot_region/"
all_files = [f for f in os.listdir(directory_path) if f.endswith("train.csv")]

# Initialize an empty list to store the dataframes
dfs = []

# Read each CSV file and append it to the list
for filename in all_files:
    if filename == "ercot_all_regions_train.csv":
        continue
    df = pd.read_csv(os.path.join(directory_path, filename))
    dfs.append(df)

# Concatenate all the dataframes together
final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv(
    f"./data/final_tables/ercot_region/ercot_all_regions_train.csv",
    index=False,
)

directory_path = f"./data/final_tables/ercot_region/"
all_files = [f for f in os.listdir(directory_path) if f.endswith("test.csv")]

# Initialize an empty list to store the dataframes
dfs = []

# Read each CSV file and append it to the list
for filename in all_files:
    if filename == "ercot_all_regions_test.csv":
        continue
    df = pd.read_csv(os.path.join(directory_path, filename))
    dfs.append(df)

# Concatenate all the dataframes together
final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv(
    f"./data/final_tables/ercot_region/ercot_all_regions_test.csv",
    index=False,
)
