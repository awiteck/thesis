import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import optuna
from optuna.trial import TrialState

std = 10011.015
mean = 43558.95


def xgboost_single_model_fit(
    horizon, n_estimators=100, learning_rate=0.1, max_depth=6, hyper_opt=False
):
    if hyper_opt:
        df_train = pd.read_csv("./data/final_tables/sarimax/train_hyper.csv")
        df_test = pd.read_csv("./data/final_tables/sarimax/test_hyper.csv")
        # df_train = pd.read_csv("/scratch/gpfs/awiteck/data/train_hyper.csv")
        # df_test = pd.read_csv("/scratch/gpfs/awiteck/data/test_hyper.csv")
    else:
        df_train = pd.read_csv("./data/final_tables/sarimax/train.csv")
        df_test = pd.read_csv("./data/final_tables/sarimax/test.csv")
        # df_train = pd.read_csv("/scratch/gpfs/awiteck/data/train.csv")
        # df_test = pd.read_csv("/scratch/gpfs/awiteck/data/test.csv")

    print("Data loaded")

    # Number of lags (hours back) to use for features
    num_lags = 24  # for example, last 24 hours

    # Initialize dictionaries to hold temporary DataFrames for lags and leads
    lag_dfs_train = {}
    lag_dfs_test = {}
    lead_dfs_train = {}
    lead_dfs_test = {}

    for lag in range(1, num_lags + 1):
        lag_dfs_train[f"Normalized Demand_lag_{lag}"] = df_train[
            "Normalized Demand"
        ].shift(lag)
        lag_dfs_test[f"Normalized Demand_lag_{lag}"] = df_test[
            "Normalized Demand"
        ].shift(lag)

    weather_variables = ["temperature", "cloudcover", "humidity", "windspeed"]
    for var in weather_variables:
        # Create lag features
        for lag in range(1, num_lags + 1):
            lag_dfs_train[f"{var}_lag_{lag}"] = df_train[var].shift(lag)
            lag_dfs_test[f"{var}_lag_{lag}"] = df_test[var].shift(lag)

        # Create lead features
        for lead in range(1, num_lags + 1):
            lead_dfs_train[f"{var}_lead_{lead}"] = df_train[var].shift(-lead)
            lead_dfs_test[f"{var}_lead_{lead}"] = df_test[var].shift(-lead)

    # Convert dictionaries to DataFrames
    lags_leads_train = pd.DataFrame(lag_dfs_train).join(pd.DataFrame(lead_dfs_train))
    lags_leads_test = pd.DataFrame(lag_dfs_test).join(pd.DataFrame(lead_dfs_test))

    df_train = pd.concat([df_train, lags_leads_train], axis=1)
    df_test = pd.concat([df_test, lags_leads_test], axis=1)

    print("Lags loaded")

    df_train = df_train.dropna().reset_index(drop=True)
    df_test = df_test.dropna().reset_index(drop=True)

    X_train = df_train.drop("Normalized Demand", axis=1)
    X_test = df_test.drop("Normalized Demand", axis=1)

    print(f"X_train.shape: {X_train.shape}")

    # Prepare y_train for each forecast horizon
    y_train = df_train["Normalized Demand"].shift(-horizon).dropna()

    print(f"y_train.shape: {y_train.shape}")

    # Truncate to ensure alignment
    X_train = X_train.iloc[:-horizon]

    y_test = df_test["Normalized Demand"]

    print("Fitting...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print("Fitted")
    # Select every 24th row starting from the 0th index (midnight) for forecasting
    midnight_indices = np.arange(0, len(X_test), 24)

    # Initialize a list to store the RMSE of each day's forecast
    daily_rmses = []

    predictions = []
    actuals = []

    # Iterate over each selected midnight index to perform 24-hour forecasts
    for start_index in midnight_indices:
        # Ensure there's enough data to forecast the next 24 hours
        if start_index + 24 > len(X_test):
            break  # Skip the last day if we don't have enough data

        predictions.append(model.predict(X_test.iloc[[start_index]])[0])
        actuals.append(y_test[start_index])

    # Convert lists to arrays for plotting and calculation
    predictions = np.array(predictions) * std + mean
    actuals = np.array(actuals) * std + mean

    # Calculate the overall RMSE across all days
    overall_rmse = np.sqrt(mean_squared_error(predictions, actuals))
    print(f"Overall RMSE for 24-hour forecasts at midnight: {overall_rmse}")

    return overall_rmse


# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# Function to calculate MAPE
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def xgboost_fit(n_estimators=100, learning_rate=0.1, max_depth=6, hyper_opt=False):
    if hyper_opt:
        df_train = pd.read_csv("./data/final_tables/sarimax/train_hyper.csv")
        df_test = pd.read_csv("./data/final_tables/sarimax/test_hyper.csv")
        # df_train = pd.read_csv("/scratch/gpfs/awiteck/data/train_hyper.csv")
        # df_test = pd.read_csv("/scratch/gpfs/awiteck/data/test_hyper.csv")
    else:
        df_train = pd.read_csv("./data/final_tables/sarimax/train.csv")
        df_test = pd.read_csv("./data/final_tables/sarimax/test.csv")
        # df_train = pd.read_csv("/scratch/gpfs/awiteck/data/train.csv")
        # df_test = pd.read_csv("/scratch/gpfs/awiteck/data/test.csv")
    print("Data loaded")

    # Number of lags (hours back) to use for features
    num_lags = 24  # for example, last 24 hours

    # Initialize dictionaries to hold temporary DataFrames for lags and leads
    lag_dfs_train = {}
    lag_dfs_test = {}
    lead_dfs_train = {}
    lead_dfs_test = {}

    for lag in range(1, num_lags + 1):
        lag_dfs_train[f"Normalized Demand_lag_{lag}"] = df_train[
            "Normalized Demand"
        ].shift(lag)
        lag_dfs_test[f"Normalized Demand_lag_{lag}"] = df_test[
            "Normalized Demand"
        ].shift(lag)

    weather_variables = ["temperature", "cloudcover", "humidity", "windspeed"]
    for var in weather_variables:
        # Create lag features
        for lag in range(1, num_lags + 1):
            lag_dfs_train[f"{var}_lag_{lag}"] = df_train[var].shift(lag)
            lag_dfs_test[f"{var}_lag_{lag}"] = df_test[var].shift(lag)

        # Create lead features
        for lead in range(1, num_lags + 1):
            lead_dfs_train[f"{var}_lead_{lead}"] = df_train[var].shift(-lead)
            lead_dfs_test[f"{var}_lead_{lead}"] = df_test[var].shift(-lead)

    # Convert dictionaries to DataFrames
    lags_leads_train = pd.DataFrame(lag_dfs_train).join(pd.DataFrame(lead_dfs_train))
    lags_leads_test = pd.DataFrame(lag_dfs_test).join(pd.DataFrame(lead_dfs_test))

    df_train = pd.concat([df_train, lags_leads_train], axis=1)
    df_test = pd.concat([df_test, lags_leads_test], axis=1)

    print("Lags loaded")

    df_train = df_train.dropna().reset_index(drop=True)
    df_test = df_test.dropna().reset_index(drop=True)

    X_train = df_train.drop("Normalized Demand", axis=1)
    X_test = df_test.drop("Normalized Demand", axis=1)

    print(f"X_train.shape: {X_train.shape}")

    # Prepare y_train for each forecast horizon
    y_trains = {
        f"y_train_{i}h": df_train["Normalized Demand"].shift(-i).dropna()
        for i in range(1, 25)
    }
    # Truncate to ensure alignment
    X_trains = {f"X_train_{i}h": X_train.iloc[:-i] for i in range(1, 25)}

    y_test = df_test["Normalized Demand"]

    models = {}

    print("Fitting...")
    for i in range(1, 25):
        # print(f"Training model for {i} hour(s) ahead...")
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_trains[f"X_train_{i}h"], y_trains[f"y_train_{i}h"])
        models[f"model_{i}h"] = model
    print("Fitted...")
    # Select every 24th row starting from the 0th index (midnight) for forecasting
    midnight_indices = np.arange(0, len(X_test), 24)

    # Initialize a list to store the RMSE of each day's forecast
    daily_rmses = []

    all_daily_predictions = []
    all_daily_actuals = []

    # Iterate over each selected midnight index to perform 24-hour forecasts
    for start_index in midnight_indices:
        # Ensure there's enough data to forecast the next 24 hours
        if start_index + 24 > len(X_test):
            break  # Skip the last day if we don't have enough data

        # Initialize a list to store the predictions for the current day
        daily_predictions = []

        # Forecast the next 24 hours using each model
        for i in range(1, 25):
            model = models[f"model_{i}h"]
            # Predict using the model and append to daily predictions
            # Note: We use the same row (start_index) for all predictions since we're forecasting the next 24 hours from midnight
            y_pred_daily = model.predict(X_test.iloc[[start_index]])
            daily_predictions.append(y_pred_daily[0])

        # Retrieve the actual values for the next 24 hours
        y_true_daily = y_test.iloc[start_index : start_index + 24].values
        # Store the daily predictions and actual values
        all_daily_predictions.extend(daily_predictions)
        all_daily_actuals.extend(y_true_daily)

        # Calculate and store the RMSE for the current day's forecast
        rmse_daily = np.sqrt(mean_squared_error(y_true_daily, daily_predictions))
        daily_rmses.append(rmse_daily)

    # Convert lists to arrays for plotting and calculation
    all_daily_predictions = np.array(all_daily_predictions) * std + mean
    all_daily_actuals = np.array(all_daily_actuals) * std + mean

    # Calculate the overall RMSE across all days
    # overall_rmse = np.mean(daily_rmses)
    # print(f"Overall RMSE for 24-hour forecasts at midnight: {overall_rmse}")

    # Calculate the overall RMSE across all days
    overall_rmse = np.sqrt(mean_squared_error(all_daily_predictions, all_daily_actuals))
    overall_mae = mean_absolute_error(all_daily_actuals, all_daily_predictions)
    overall_mape = mape(all_daily_actuals, all_daily_predictions)
    print(f"Overall RMSE for 24-hour forecasts at midnight: {overall_rmse}")
    print(f"Overall MAE for 24-hour forecasts at midnight: {overall_mae}")
    print(f"Overall MAPE for 24-hour forecasts at midnight: {overall_mape}")

    return overall_rmse

    """
    y_preds = {}
    for i in range(1, 25):
        print(f"Forecasting {i} hour(s) ahead...")
        model = models[f"model_{i}h"]
        y_preds[f"y_pred_{i}h"] = model.predict(X_test)

    for i in range(1, 25):
        # Ensuring that the lengths are the same by trimming the excess predictions
        valid_length = len(y_tests[f"y_test_{i}h"])
        y_pred_trimmed = y_preds[f"y_pred_{i}h"][:valid_length]
        y_test_trimmed = y_tests[f"y_test_{i}h"]

        # Now, y_pred_trimmed and y_test_trimmed can be directly compared
        rmse = np.sqrt(mean_squared_error(y_test_trimmed, y_pred_trimmed))
        print(f"RMSE for {i} hours ahead: {rmse}")
    """


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

    df = df[df["year"] <= 2022]

    # Locate rows where the 'year' column is 2023 and change them to 2022
    df.loc[df["year"] == 2022, "year"] = 2021

    target_date = pd.to_datetime("2022-01-01") - pd.Timedelta(hours=24)
    df["2022"] = 0
    df.loc[(df["Region"] == "erco") & (df["timestamp"] > target_date), "2022"] = 1

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

    df_train = df[df["2022"] == 0]
    df_test = df[df["2022"] == 1]

    df_train = df_train[input_vars].astype("float32")
    df_test = df_test[input_vars].astype("float32")

    df_train.to_csv(
        f"./data/final_tables/sarimax/train_hyper.csv",
        index=False,
    )
    df_test.to_csv(
        f"./data/final_tables/sarimax/test_hyper.csv",
        index=False,
    )


def hyperparam_opt():
    def objective(trial):
        # Define the range of hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        max_depth = trial.suggest_int("max_depth", 1, 10)
        # Train and evaluate the model
        performance_metric = xgboost_fit(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            hyper_opt=True,
        )
        return performance_metric

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=2, interval_steps=1
            ),
            patience=1,
        ),
        sampler=sampler,
    )
    study.optimize(objective, n_trials=50)

    # After the optimization is done
    best_params = study.best_params

    # Convert the best_params dict to a string for writing to file
    best_params_str = "\n".join(
        [f"{key}: {value}" for key, value in best_params.items()]
    )

    # Write the best hyperparameters to a file
    with open(f"xgboost_best_params.txt", "w") as f:
        f.write(best_params_str)

    print("Best hyperparameters: ", study.best_params)
    print(f"Best hyperparameters saved to xgboost_best_params.txt.txt")


def hyperparam_opt_single_model(horizon):
    def objective(trial):
        # Define the range of hyperparameters
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        max_depth = trial.suggest_int("max_depth", 1, 10)
        # Train and evaluate the model
        performance_metric = xgboost_single_model_fit(
            horizon,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            hyper_opt=True,
        )
        return performance_metric

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=2, interval_steps=1
            ),
            patience=1,
        ),
        sampler=sampler,
    )
    study.optimize(objective, n_trials=50)

    # After the optimization is done
    best_params = study.best_params

    # Convert the best_params dict to a string for writing to file
    best_params_str = "\n".join(
        [f"{key}: {value}" for key, value in best_params.items()]
    )

    # Write the best hyperparameters to a file
    with open(f"xgboost_best_params_h{horizon}.txt", "w") as f:
        f.write(best_params_str)

    print("Best hyperparameters: ", study.best_params)
    print(f"Best hyperparameters saved to xgboost_best_params_h{horizon}.txt")


def hyperparam_opt_all_models():
    for horizon in range(24):
        hyperparam_opt_single_model(horizon)


# hyperparam_opt()

# hyperparam_opt_all_models()
xgboost_fit(n_estimators=904, learning_rate=0.109, max_depth=5, hyper_opt=False)

# process_data()
