from config import get_config
from basic_train_helper import run_training_and_validation
import argparse
from basic_erco_2023 import test_2023

config = get_config()

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--T", type=int, help="T")
parser.add_argument("--s", type=int, help="Step size")
args = parser.parse_args()

config["data_path"] = "/scratch/gpfs/awiteck/data/composite.csv"
config["enc_seq_len"] = args.T
config["output_sequence_length"] = 24
config["step_size"] = args.s
config["batch_size"] = 32
config["lr"] = 3e-4
config["loss_fn"] = "huber"
config["num_epochs"] = 100


config["experiment_name"] = f"basic_linear_s{args.s}_T{config['enc_seq_len']}"

run_training_and_validation(
    config, write_example_predictions=False, small_dataset=False, shuffle=False
)
test_2023(experiment_name=config["experiment_name"], T=config["enc_seq_len"])
