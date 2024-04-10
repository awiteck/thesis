from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


def process_data():
    def normalize_continuous_vars(data, var_names):
        # mean/std normalize.
        for var in var_names:
            assert (
                var in data.columns
            ), f"Error: {var} is not a column of input dataframe."
            # Z-score normalization
            mean = data[var].mean()
            std = data[var].std()
            print(f"{var}: mean: {mean}, std: {std}")
            data.loc[:, var] = (data[var] - mean) / std
        return data

    df = pd.read_csv(
        "./data/final_tables/erco_2023_andcomposite/erco_2023_andcomposite.csv"
    )

    print(f"Data retrieved.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df[df["year"] <= 2023]
    num_rows = (df["year"] == 2023).sum()

    # Locate rows where the 'year' column is 2023 and change them to 2022
    df.loc[df["year"] == 2023, "year"] = 2022
    df.loc[df["Region"] == "erco_2023", "Region"] = "erco"

    target_date = pd.to_datetime("2023-01-01") - pd.Timedelta(hours=24)
    df.loc[(df["Region"] == "erco") & (df["timestamp"] > target_date), "2023"] = 1

    # Ensure all continuous vars are normalized
    df = normalize_continuous_vars(
        data=df, var_names=["temperature", "cloudcover", "humidity", "windspeed"]
    )
    print("Continuous data normalized.")

    weather_columns = ["temperature", "cloudcover", "humidity", "windspeed"]

    for col in weather_columns:
        df[f"{col}_24h_ahead"] = df.groupby("Region")[col].shift(-24)

    df = df.dropna(
        subset=[
            "windspeed_24h_ahead",
            "temperature_24h_ahead",
            "cloudcover_24h_ahead",
            "humidity_24h_ahead",
        ]
    )

    # One-hot encode the 'month', 'day_of_week', 'hour', and 'Region' columns
    categorical_columns = ["year", "month", "day", "day_of_week", "hour", "Region"]
    # Step 1: Before encoding, retrieve the unique values for each categorical variable
    unique_values_per_categorical_column = {
        col: df[col].unique() for col in categorical_columns
    }
    df = pd.get_dummies(df, columns=categorical_columns)

    # Step 3: Construct the expected new column names for each categorical variable
    input_vars = [
        "Normalized Demand",
        "temperature",
        "cloudcover",
        "humidity",
        "windspeed",
        "temperature_24h_ahead",
        "cloudcover_24h_ahead",
        "humidity_24h_ahead",
        "windspeed_24h_ahead",
    ]
    for col, unique_values in unique_values_per_categorical_column.items():
        # Sort the unique values to maintain order; this step may be adjusted based on specific needs
        sorted_unique_values = sorted(unique_values)
        for val in sorted_unique_values:
            new_col_name = f"{col}_{val}"
            if (
                new_col_name in df.columns
            ):  # Ensure the column name exists in the DataFrame
                input_vars.append(new_col_name)

    df_train = df[df["2023"] == 0]
    df_test = df[df["2023"] == 1]

    df_train = df_train[input_vars].astype("float32")
    df_test = df_test[input_vars].astype("float32")

    df_train.to_csv(
        f"./data/final_tables/sarimax/train.csv",
        index=False,
    )
    df_test.to_csv(
        f"./data/final_tables/sarimax/test.csv",
        index=False,
    )


def sarimax_fit():
    df_train = pd.read_csv("./data/final_tables/sarimax/train.csv")[:10000]
    df_test = pd.read_csv("./data/final_tables/sarimax/test.csv")[:1000]

    # Target variable
    y_train = df_train["Normalized Demand"]

    # Exogenous variables (excluding 'Normalized Demand')
    exog_train = df_train.drop(columns=["Normalized Demand"])

    print("Train data loaded.")
    # Building the model with example parameters
    # Example: AR(2) model with exogenous variables, no differencing or MA component
    model = SARIMAX(
        endog=y_train, exog=exog_train, order=(24, 0, 0), seasonal_order=(0, 0, 0, 0)
    )
    print("Model constructed. Fitting...")

    # Fitting the model
    model_fit = model.fit(disp=False)

    print("Fitted. Testing...")

    exog_test = df_test.drop(columns=["Normalized Demand"], errors="ignore")

    # Forecasting
    # The number of steps to forecast ('steps') should match the length of your test data
    forecast = model_fit.get_forecast(steps=len(df_test), exog=exog_test)

    print("Test forecast done.")

    # The forecast result object contains various attributes, such as predicted mean
    forecast_mean = forecast.predicted_mean

    print(f"forecast_mean: {forecast_mean}")


sarimax_fit()
"""

# Step 3: Construct the expected new column names for each categorical variable
input_vars = [
    "Normalized Demand",
    "temperature",
    "cloudcover",
    "humidity",
    "windspeed",
    "temperature_24h_ahead",
    "cloudcover_24h_ahead",
    "humidity_24h_ahead",
    "windspeed_24h_ahead",
]
for col, unique_values in unique_values_per_categorical_column.items():
    # Sort the unique values to maintain order; this step may be adjusted based on specific needs
    sorted_unique_values = sorted(unique_values)
    for val in sorted_unique_values:
        new_col_name = f"{col}_{val}"
        if new_col_name in df.columns:  # Ensure the column name exists in the DataFrame
            input_vars.append(new_col_name)
"""

# data = df
# data = data[input_vars].astype("float32")

# # Ensure all continuous vars are normalized
# data = utils.normalize_discrete_vars(data=data, var_names=config["discrete_vars"])
# print("Discrete data normalized.")

# # Calculate discrete variable dimensions and corresponding embedding dimensions
# discrete_var_dims = utils.calculate_discrete_dims(data, config["discrete_vars"])

# print(f"discrete_vars: {config['discrete_vars']}")
# print(f"discrete_var_dims: {discrete_var_dims}")


# config["discrete_var_dims"] = discrete_var_dims
# config["discrete_embedding_dims"] = [
#     round(1.6 * math.sqrt(dim)) for dim in discrete_var_dims
# ]

# data_2023 = data[data["2023"] == 1]
# # data_2023 = data[(data["Region"] == "erco") & (data["timestamp"] > target_date)]

# data = data_2023

# #
# #
# #
# #
# #
# #

# #
# #
# weather_columns = ["temperature", "cloudcover", "humidity", "windspeed"]

# # Ensure all continuous vars are normalized
# df = utils.normalize_continuous_vars(
#     data=df, var_names=["temperature", "cloudcover", "humidity", "windspeed"]
# )
# print("Continuous data normalized.")

# for col in weather_columns:
#     df[f"{col}_24h_ahead"] = df.groupby("Region")[col].shift(-24)

# df = df.dropna(
#     subset=[
#         "windspeed_24h_ahead",
#         "temperature_24h_ahead",
#         "cloudcover_24h_ahead",
#         "humidity_24h_ahead",
#     ]
# )

# # One-hot encode the 'month', 'day_of_week', 'hour', and 'Region' columns
# categorical_columns = ["year", "month", "day", "day_of_week", "hour", "Region"]
# # Step 1: Before encoding, retrieve the unique values for each categorical variable
# unique_values_per_categorical_column = {
#     col: df[col].unique() for col in categorical_columns
# }
# df = pd.get_dummies(df, columns=categorical_columns)

# # Step 3: Construct the expected new column names for each categorical variable
# input_vars = [
#     "Normalized Demand",
#     "temperature",
#     "cloudcover",
#     "humidity",
#     "windspeed",
#     "temperature_24h_ahead",
#     "cloudcover_24h_ahead",
#     "humidity_24h_ahead",
#     "windspeed_24h_ahead",
# ]
# for col, unique_values in unique_values_per_categorical_column.items():
#     # Sort the unique values to maintain order; this step may be adjusted based on specific needs
#     sorted_unique_values = sorted(unique_values)
#     for val in sorted_unique_values:
#         new_col_name = f"{col}_{val}"
#         if new_col_name in df.columns:  # Ensure the column name exists in the DataFrame
#             input_vars.append(new_col_name)

# data = df
# data = data[input_vars].astype("float32")
