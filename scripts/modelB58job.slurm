#!/bin/bash
#SBATCH --job-name=B58models       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4              # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=05:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fault
#SBATCH --mail-user=awiteck@princeton.edu

module purge
module load anaconda3/2023.9
conda activate transformer-env

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun -N 1 -n 1 --gres=gpu:1 --exclusive python -u trainval.py --letter B --number 5 &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python -u trainval.py --letter B --number 6 &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python -u trainval.py --letter B --number 7 &
srun -N 1 -n 1 --gres=gpu:1 --exclusive python -u trainval.py --letter B --number 8 &
wait