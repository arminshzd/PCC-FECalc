#!/bin/sh
##SBATCH --job-name=PCC

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# name of partition to queue on
##SBATCH --account=pi-andrewferguson
##SBATCH --partition=andrewferguson-gpu
#SBATCH --partition=gm4-pmext
#SBATCH --qos=gm4-cpu

# max wall time for job (HH:MM:SS)
#SBATCH --time=1-12:00:00

# number of nodes for this job
#SBATCH --nodes=1

# number of processes to run per node
#SBATCH --ntasks-per-node=8

# number of threads per cpu
#SBATCH --cpus-per-task=5

# reserve the specified node(s) for this job
#SBATCH --exclusive

NCPU=$(($SLURM_NTASKS_PER_NODE))
NTHR=$(($SLURM_CPUS_PER_TASK))
NNOD=$(($SLURM_JOB_NUM_NODES))

NP=$(($NCPU * $NNOD))

/project2/andrewferguson/armin/wham/wham-2d/wham-2d Px=0 0.2 3.0 100 Py=pi -3.1416 3.1416 500 0.001 300 2 metadata.dat fdat.dat 0
