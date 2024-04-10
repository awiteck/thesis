from config import get_config
from train_helper import run_training_and_validation
import argparse

cfg = get_config()

letter_map = {
    "A": [],
    "B": [
        "direct_radiation_24h_ahead_5p",
        "diffuse_radiation_24h_ahead_5p",
        "windspeed_24h_ahead_5p",
    ],
    "C": [
        "direct_radiation_24h_ahead_10p",
        "diffuse_radiation_24h_ahead_10p",
        "windspeed_24h_ahead_10p",
    ],
    "D": [
        "direct_radiation_24h_ahead_20p",
        "diffuse_radiation_24h_ahead_20p",
        "windspeed_24h_ahead_20p",
    ],
    "E": [
        "direct_radiation_24h_ahead_50p",
        "diffuse_radiation_24h_ahead_50p",
        "windspeed_24h_ahead_50p",
    ],
    "F": [
        "direct_radiation_24h_ahead_100p",
        "diffuse_radiation_24h_ahead_100p",
        "windspeed_24h_ahead_100p",
    ],
}


cfg["dropout"] = 0.041  # 0.061
cfg["lr"] = 3.8e-4  # 2.9e-04
cfg["batch_size"] = 32
cfg["d_model"] = 32  # 256
cfg["N"] = 1  # 4
cfg["enc_seq_len"] = 12  # 72
cfg["output_sequence_length"] = 24
cfg["step_size"] = 12 + 24  # 1


cfg["num_epochs"] = 150
cfg["predicted_features"] = ["load", "wind", "solar"]
# cfg["predicted_features"] = ["wind", "solar"]
cfg["continuous_vars"] = [
    "temperature",
    "cloudcover",
    "windspeed",
    "direct_radiation",
    "diffuse_radiation",
]
cfg["discrete_vars"] = [
    "region",
    "year",
    "month",
    "day_of_week",
    "day",
    "hour",
]
cfg["data_path"] = "/scratch/gpfs/awiteck/data/ercot_all_regions_train.csv"


parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--letter", type=str, help="Letter of the experiment")

# Parse the arguments
args = parser.parse_args()

if args.letter:
    cfg["continuous_vars"] = cfg["continuous_vars"] + letter_map[args.letter]
else:
    print("No model number provided.")
cfg["experiment_name"] = f"wind_solar_{args.letter}_2024_03_16"


run_training_and_validation(cfg, write_example_predictions=True)
