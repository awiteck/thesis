#!/bin/bash
#SBATCH --job-name=basic_train    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=awiteck@princeton.edu

module purge
module load anaconda3/2023.9
conda activate transformer-env

python -u basic_trainval.py --T 24 --s 2
python -u basic_trainval.py --T 24 --s 3
python -u basic_trainval.py --T 24 --s 6
python -u basic_trainval.py --T 24 --s 12
python -u basic_trainval.py --T 24 --s 24
python -u basic_trainval.py --T 24 --s 48