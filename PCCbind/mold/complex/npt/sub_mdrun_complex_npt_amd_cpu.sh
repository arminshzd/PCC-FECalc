#!/bin/sh

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# name of partition to queue on
#SBATCH --account=pi-andrewferguson
#SBATCH --partition=amd

# max wall time for job (HH:MM:SS)
#SBATCH --time=1-12:00:00

# number of GPU(s) per node, if available
#SBATCH --gres=gpu:0

# number of nodes for this job
#SBATCH --nodes=1

# specific nodes
#SBATCH --exclude=midway3-05[01-02,09-10,15-17,23,26,28-31,33-39,49]

# number of processes to run per node
#SBATCH --ntasks-per-node=8

# number of threads per cpu
#SBATCH --cpus-per-task=5

# reserve the specified node(s) for this job
#SBATCH --exclusive

NCPU=$(($SLURM_NTASKS_PER_NODE))
NTHR=$(($SLURM_CPUS_PER_TASK))
NNOD=$(($SLURM_JOB_NUM_NODES))

NP=$(($NCPU * $NNOD * $NTHR))

module unload openmpi gcc cuda python
module load gcc/10.2.0 openmpi gsl

source /project/andrewferguson/armin/grom_new/gromacs-2021.6/installed-files-mw3-amd/bin/GMXRC

gmx grompp -f npt.mdp -c ../nvt/nvt.gro -r ../nvt/nvt.gro -t ../nvt/nvt.cpt -p topol.top -o npt.tpr

gmx mdrun -ntomp "$NP" -deffnm npt

