import pandas as pd


def create_multiregion_table(region_names):
    dfs = []
    for region_name in region_names:
        df_path = f"./data/final_tables/{region_name}/{region_name}.csv"
        df = pd.read_csv(df_path)
        dfs.append(df)
    final_df = pd.concat(dfs, ignore_index=True)

    # Generate the region_to_id dictionary
    region_to_id = {name: idx for idx, name in enumerate(region_names)}
    # Use map to create the new column
    final_df["region_id"] = final_df["Region"].map(region_to_id)
    final_df.to_csv(f"./data/final_tables/composite/composite.csv", index=False)
    return final_df


region_names = ["banc", "erco", "isne"]
create_multiregion_table(region_names)
