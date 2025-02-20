#!/bin/sh

# output file (including stderr)
#SBATCH --output=MOBO.log

# name of partition to queue on
#SBATCH --partition=gm4-pmext
#SBATCH --qos=gm4-cpu

# max wall time for job (HH:MM:SS)
#SBATCH --time=1-12:00:00

# number of GPU(s) per node, if available
##SBATCH --gres=gpu:0

# number of nodes for this job
#SBATCH --nodes=1

# number of processes to run per node
#SBATCH --ntasks-per-node=10

# number of threads per cpu
#SBATCH --cpus-per-task=1

NCPU=$(($SLURM_NTASKS_PER_NODE))
NTHR=$(($SLURM_CPUS_PER_TASK))
NNOD=$(($SLURM_JOB_NUM_NODES))

NP=$(($NCPU * $NNOD * $NTHR))

module load python/anaconda-2021.05

source activate /project/andrewferguson/armin/envs/torch2

python get_cands.py >> "$1".out
