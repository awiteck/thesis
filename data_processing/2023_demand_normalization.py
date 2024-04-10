import pandas as pd


def normalize_continuous_vars(data, var_names):
    # mean/std normalize.
    for var in var_names:
        assert var in data.columns, f"Error: {var} is not a column of input dataframe."
        # Z-score normalization
        mean = data[var].mean()
        std = data[var].std()
        print(f"mean: {mean}")
        print(f"std: {std}")
        data.loc[:, var] = (data[var] - mean) / std
    return data


df = pd.read_csv("./data/final_tables/banc2023andcomposite/banc2023andcomposite.csv")
data = normalize_continuous_vars(data=df, var_names=["Normalized Demand"])
