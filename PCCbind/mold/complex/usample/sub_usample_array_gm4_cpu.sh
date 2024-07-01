#!/bin/sh

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# array setup
#SBATCH --array=0-19
# name of partition to queue on
#SBATCH --partition=gm4-pmext
#SBATCH --qos=gm4-cpu

# max wall time for job (HH:MM:SS)
#SBATCH --time=1-12:00:00

# number of GPU(s) per node, if available
#SBATCH --gres=gpu:0

# number of nodes for this job
#SBATCH --nodes=1

# number of processes to run per node
#SBATCH --ntasks-per-node=8

# number of threads per cpu
#SBATCH --cpus-per-task=5

NCPU=$(($SLURM_NTASKS_PER_NODE))
NTHR=$(($SLURM_CPUS_PER_TASK))
NNOD=$(($SLURM_JOB_NUM_NODES))

NP=$(($NCPU * $NNOD * $NTHR))

module unload openmpi gcc cuda python
module load openmpi/4.1.1+gcc-10.1.0 cuda/11.2

source /project/andrewferguson/armin/grom_new/gromacs-2021.6/installed-files-mw2-256/bin/GMXRC

echo window "$SLURM_ARRAY_TASK_ID":
echo "$SLURMD_NODENAME"
cd ./runs/"$SLURM_ARRAY_TASK_ID"/

if [ -f ./.done ]; then
    exit 1
fi

if [ -f ./md2.gro ]; then
    exit 1
fi


if [ -f ./md1.cpt ]; then
    gmx mdrun -ntomp "$NP" -s md1.tpr -cpi md1.cpt -deffnm md1 -plumed plumed.dat
else
    gmx grompp -f md1.mdp -c ../../../npt/npt.gro -r ../../../npt/npt.gro -t ../../../npt/npt.cpt -p topol.top -o md1.tpr
    gmx mdrun -ntomp "$NP" -deffnm md1 -plumed plumed.dat
fi

if [ ! -f ./md2.log ]; then
    rm ./us.dat
fi

if [ -f ./md2.cpt ]; then
    gmx mdrun -ntomp "$NP" -s md2.tpr -cpi md2.cpt -deffnm md2 -plumed plumed.dat
else
    gmx grompp -f md2.mdp -c md1.gro -r md1.gro -t md1.cpt -p topol.top -o md2.tpr
    gmx mdrun -ntomp "$NP" -deffnm md2 -plumed plumed.dat
fi
cd ../../