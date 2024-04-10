import os
import pandas as pd
import numpy as np


def get_concatenated_table(region="banc_2023"):
    """
    Given an arbitrary number of small tables downloaded from eia.gov,
    this concatenates all of them and sorts by time, retuning a final
    concatenated table.
    """

    path1 = f"./data/final_tables/composite/composite.csv"
    path2 = f"./data/final_tables/{region}/{region}.csv"

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    df1["2023"] = 0
    df2["2023"] = 1
    dfs = [df1, df2]

    # Concatenate all the dataframes together
    final_df = pd.concat(dfs, ignore_index=True)
    df_filtered = final_df[final_df["year"] > 2015]

    df_filtered.to_csv(
        f"./data/final_tables/{region}_andcomposite/{region}_andcomposite.csv",
        index=False,
    )
    return final_df


df = get_concatenated_table(region="isne_2023")
print(df.head())
