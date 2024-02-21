from config import get_config
from train_helper import run_training_and_validation
import argparse

cfg = get_config()
cfg["num_epochs"] = 100
cfg["patience"] = 10
cfg["loss_fn"] = "huber"
cfg["N"] = 2
cfg["dropout"] = 0.1
cfg["data_path"] = "/scratch/network/awiteck/data/gefcom12.csv"
cfg["target_col_name"] = "load_normalized"
cfg["continuous_vars"] = []
# cfg["discrete_vars"] = ["hour", "zone_id"]
# cfg["discrete_vars"] = []
cfg["discrete_vars"] = ["year", "month", "day", "hour", "is_holiday", "zone_id"]
cfg["experiment_name"] = "2024_2_21_gefcom_allvars"

# Create the parser
parser = argparse.ArgumentParser(description="Training script")
# Add an argument that accepts a list of values
parser.add_argument("--continuous_vars", nargs="*", help="List of continuous variables")
# Add an argument that accepts a list of values
parser.add_argument("--contextual_vars", nargs="*", help="List of contextual variables")


# Parse the arguments
args = parser.parse_args()

# Use the argument
if args.continuous_vars:
    print("Received continuous variables:")
    for var in args.continuous_vars:
        print(var)
    cfg["continuous_vars"] = args.continuous_vars
else:
    print("No contextual variables provided.")

# Use the argument
if args.contextual_vars:
    print("Received contextual variables:")
    for var in args.contextual_vars:
        print(var)
    cfg["contextual_vars"] = args.contextual_vars
else:
    print("No contextual variables provided.")

run_training_and_validation(cfg, write_example_predictions=True)
