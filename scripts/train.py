from config import get_config
from train_helper import train_model
import argparse

cfg = get_config()
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

train_model(cfg)
