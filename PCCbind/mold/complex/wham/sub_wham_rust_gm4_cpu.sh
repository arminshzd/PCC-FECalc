#!/bin/sh

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# name of partition to queue on
#SBATCH --partition=gm4-pmext
#SBATCH --qos=gm4-cpu

# max wall time for job (HH:MM:SS)
#SBATCH --time=1-12:00:00

# number of nodes for this job
#SBATCH --nodes=1

# number of processes to run per node
#SBATCH --ntasks-per-node=4

# number of threads per cpu
#SBATCH --cpus-per-task=5

# reserve the specified node(s) for this job
##SBATCH --exclusive

NCPU=$(($SLURM_NTASKS_PER_NODE))
NTHR=$(($SLURM_CPUS_PER_TASK))
NNOD=$(($SLURM_JOB_NUM_NODES))

NP=$(($NCPU * $NNOD))

/home/arminsh/.cargo/bin/wham -v --bins 100 --max 4.2 --file ../usample/runs/metadata.dat --min 0.0  --temperature 300.0 --bt 10
