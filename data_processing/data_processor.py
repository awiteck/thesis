import demand_helper as demand
import weather_helper as weather
import numpy as np
import pandas as pd
import holidays


def create_table(region_name, coords, mean_demand=None, std_demand=None, timezone=None):
    # Get all available electric load data for the region first
    load_data = demand.get_concatenated_table(
        region_name, mean_demand=mean_demand, std_demand=std_demand
    )

    print(load_data.head())

    time_start = load_data["timestamp"].min().strftime("%Y-%m-%d")
    time_end = load_data["timestamp"].max().strftime("%Y-%m-%d")

    # Initialize an empty DataFrame to hold the merged data
    weather_data = pd.DataFrame()

    for i, coord in enumerate(coords):
        # Retrieve the weather data for the current region
        small_weather_df = weather.getWeatherData(
            coord[0], coord[1], time_start, time_end, timezone
        )

        # Rename the columns to indicate the region, skip renaming the 'Time' column
        region_columns = {
            col: f"{col}_{i+1}" for col in small_weather_df.columns if col != "Time"
        }
        small_weather_df.rename(columns=region_columns, inplace=True)

        # Merge with the main DataFrame on 'Time'
        if weather_data.empty:
            weather_data = small_weather_df
        else:
            weather_data = pd.merge(
                weather_data, small_weather_df, on="Time", how="outer"
            )

    # The final DataFrame will have columns for each region's weather data, joined on 'Time'
    print(weather_data.head())

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
    df_merged["year"] = df_merged["timestamp"].dt.year
    df_merged["month"] = df_merged["timestamp"].dt.month
    df_merged["day_of_week"] = df_merged["timestamp"].dt.dayofweek
    df_merged["day"] = df_merged["timestamp"].dt.day
    df_merged["hour"] = df_merged["timestamp"].dt.hour

    discrete_columns_of_interest = ["year", "month", "day_of_week", "day", "hour"]
    for column in discrete_columns_of_interest:
        df_merged[column] = df_merged[column] - min(df_merged[column])

    continuous_columns_of_interest = [
        "temperature",
        "humidity",
        "cloudcover",
        "windspeed",
        "precipitation",
    ]
    for column in df_merged.columns:
        for base_name in continuous_columns_of_interest:
            if (
                base_name in column
            ):  # This checks if the base column name is in the actual column name
                # Now, normalize this specific column
                mean = np.mean(df_merged[column])
                std = np.std(df_merged[column])
                df_merged[column] = (df_merged[column] - mean) / std
                print(f"{column} -- mean: {mean}, std: {std}")

    us_holidays = holidays.UnitedStates()
    # Create a new column in your DataFrame to indicate if a date is a holiday
    df_merged["is_holiday"] = df_merged["timestamp"].apply(
        lambda x: 1 if x in us_holidays else 0
    )
    # df_merged["2023"] = 1
    # df_filtered = df_merged[df_merged["year"] > 2015]
    # df_filtered = df_merged[df_merged["year"] < 2023]

    target_date = pd.to_datetime("2023-01-01") - pd.Timedelta(hours=24)

    # df.loc[(df["Region"] == "erco") & (df["timestamp"] > target_date), "2023"] = 1

    # df_merged["timestamp"] >= pd.Timedelta(hours=24)

    df_train = df_merged[(df_merged["year"] > 0) & (df_merged["year"] < 8)]
    # df_train = df_train[df_train["year"] < 2023]

    # df_test = df_merged[df_merged["year"] >= 8]
    df_test = df_merged[df_merged["timestamp"] > target_date]
    df_test = df_test[df_test["year"] < 9]
    print(df_train.head())
    print(df_test.head())

    df_hyperparam_train = df_merged[(df_merged["year"] > 0) & (df_merged["year"] < 7)]
    df_hyperparam_test = df_merged[df_merged["year"] == 7]

    # print(f"final mean: {df_merged['Demand (MWh)'].mean()}")
    # print(f"final std: {df_merged['Demand (MWh)'].std()}")

    df_train.to_csv(
        f"./data/final_tables/{region_name}/{region_name}_train.csv", index=False
    )
    df_test.to_csv(
        f"./data/final_tables/{region_name}/{region_name}_test.csv", index=False
    )
    df_hyperparam_train.to_csv(
        f"./data/final_tables/{region_name}/{region_name}_hyperparam_train.csv",
        index=False,
    )
    df_hyperparam_test.to_csv(
        f"./data/final_tables/{region_name}/{region_name}_hyperparam_test.csv",
        index=False,
    )

    # df_filtered.to_csv(
    #     f"./data/final_tables/{region_name}/{region_name}.csv", index=False
    # )
    # return df_filtered


# df = create_table(
#     "banc",
#     38.5816,
#     -121.489906,
#     timezone="America/Los_Angeles",
# )
# print(df.head())

# df = create_table(
#     "isne",
#     43.1026,
#     -71.6165,
#     timezone="America/New_York",
# )
# print(df.head())

df = create_table(
    "erco",
    [
        [29.7604, -95.3698],
        [29.4252, -98.4946],
        [32.7767, -96.7970],
        [26.2034, -98.2300],
        [27.8006, -97.3964],
        [31.9973, -102.0779],
    ],
    timezone="America/Chicago",
)

# df = create_table(
#     "banc_2023",
#     38.5816,
#     -121.489906,
#     mean_demand=1990.813,
#     std_demand=530.941,
#     timezone="America/Los_Angeles",
# )
# print(df.head())

# df = create_table(
#     "erco_2023",
#     30.5434,
#     -97.1943,
#     mean_demand=43558.95,
#     std_demand=10011.015,
#     timezone="America/Chicago",
# )
# print(df.head())


# df = create_table(
#     "isne_2023",
#     43.1026,
#     -71.6165,
#     mean_demand=13664.742,
#     std_demand=2578.648,
#     timezone="America/New_York",
# )
# print(df.head())
