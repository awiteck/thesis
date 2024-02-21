import pandas as pd
from AR import AR


def train_and_validate(path, p=24, num_preds=12):
    print("loading data...")
    df = pd.read_csv(path)
    data = df["Normalized Demand"]
    print("data loaded...")
    split = int(0.9 * len(data))
    data_train = data[:split]
    data_test = data[split:]

    ar = AR(p)
    print("AR constructed. Fitting...")
    ar.fit(data_train)
    print("Fitted. Validating...")
    rmse = ar.validate(data_test, num_preds=num_preds)
    print(f"average rmse: {rmse}")
    return rmse


path = "./data/final_tables/banc/banc.csv"
train_and_validate(path)
