from config import get_config
from train_helper import run_training_and_validation
import argparse

cfg = get_config()

# MODEL C
# cfg["N"] = 3
# cfg["d_model"] = 512
# cfg["lr"] = 2.8e-04
# cfg["dropout"] = 0.051
# cfg["num_epochs"] = 150

# MODEL A
cfg["N"] = 2
cfg["d_model"] = 64
cfg["lr"] = 4.4e-03
cfg["dropout"] = 0.047
cfg["num_epochs"] = 150


# cfg["continuous_vars"] = []
# cfg["discrete_vars"] = []
# Create the parser
parser = argparse.ArgumentParser(description="Training script")

# parser.add_argument("--continuous_vars", nargs="*", help="List of continuous variables")
# parser.add_argument("--discrete_vars", nargs="*", help="List of discrete variables")
# parser.add_argument("--experiment_name", type=str, help="Name of the experiment")

parser.add_argument("--letter", type=str, help="Letter of the experiment")
parser.add_argument("--number", type=str, help="Number of the experiment")
parser.add_argument("--s", type=int, help="Step size of the experiment")

letter_to_length = {
    "A": {"input_length": 24, "output_length": 24},
    "B": {"input_length": 24, "output_length": 12},
    "C": {"input_length": 48, "output_length": 24},
    "D": {"input_length": 36, "output_length": 12},
    "E": {"input_length": 36, "output_length": 24},
    "F": {"input_length": 168, "output_length": 24},
}
number_to_var = {
    "1": {"continuous_vars": [], "discrete_vars": ["Region"]},
    "2": {"continuous_vars": [], "discrete_vars": ["Region", "day_of_week", "hour"]},
    "3": {
        "continuous_vars": [],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "4": {
        "continuous_vars": ["temperature"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "5": {
        "continuous_vars": ["temperature", "humidity"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "6": {
        "continuous_vars": ["temperature", "windspeed"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "7": {
        "continuous_vars": ["temperature", "cloudcover"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "8": {
        "continuous_vars": ["temperature", "humidity", "windspeed"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "9": {
        "continuous_vars": ["temperature", "windspeed", "cloudcover"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "10": {
        "continuous_vars": ["temperature", "humidity", "cloudcover"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
    "11": {
        "continuous_vars": ["temperature", "humidity", "windspeed", "cloudcover"],
        "discrete_vars": ["Region", "year", "month", "day", "day_of_week", "hour"],
    },
}

# Parse the arguments
args = parser.parse_args()

if args.letter:
    cfg["enc_seq_len"] = letter_to_length[args.letter]["input_length"]
    cfg["output_sequence_length"] = letter_to_length[args.letter]["output_length"]
else:
    print("No model letter provided.")
if args.number:
    cfg["continuous_vars"] = number_to_var[args.number]["continuous_vars"]
    cfg["discrete_vars"] = number_to_var[args.number]["discrete_vars"]
else:
    print("No model number provided.")

if args.s:
    cfg["step_size"] = args.s
cfg["experiment_name"] = f"{args.letter}{args.number}_s{args.s}_2024_04_01"

# Use the argument
# if args.continuous_vars:
#     print("Received continuous variables:")
#     for var in args.continuous_vars:
#         print(var)
#     cfg["continuous_vars"] = args.continuous_vars
# else:
#     print("No contextual variables provided.")

# # Use the argument
# if args.discrete_vars:
#     print("Received discrete variables:")
#     for var in args.discrete_vars:
#         print(var)
#     cfg["discrete_vars"] = args.discrete_vars
# else:
#     print("No discrete variables provided.")

#     # Use the argument
# if args.experiment_name:
#     print(f"Experiment name: {args.experiment_name}")
#     cfg["experiment_name"] = args.experiment_name
# else:
#     print("No experiment name provided.")

run_training_and_validation(
    cfg, write_example_predictions=True, small_dataset=False, shuffle=False
)
