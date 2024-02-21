from config import get_config
from train_helper import run_training_and_validation
import argparse

cfg = get_config()
cfg["num_epochs"] = 2
cfg["batch_size"] = 32
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
