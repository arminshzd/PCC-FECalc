#!/bin/sh
##SBATCH --job-name=PCC

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# email on start, end, and abortion
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arminsh@uchicago.edu

# name of partition to queue on
##SBATCH --account=pi-andrewferguson
##SBATCH --partition=andrewferguson-gpu
#SBATCH --partition=gm4-pmext
#SBATCH --qos=gm4

# max wall time for job (HH:MM:SS)
#SBATCH --time=1-12:00:00

# number of GPU(s) per node, if available
#SBATCH --gres=gpu:4

# number of nodes for this job
#SBATCH --nodes=2

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

module unload openmpi gcc cuda python
#module load openmpi/4.1.1 gcc/7.4.0 cuda/11.2
module load openmpi/4.1.1+gcc-10.1.0 cuda/11.2

#source /project/andrewferguson/armin/grom_new/gromacs-2021.6/installed-files-nompi/bin/GMXRC
source /project/andrewferguson/armin/grom_new/gromacs-2021.6/installed-files-mw2/bin/GMXRC

gmx grompp -f md.mdp -c ../npt/npt.gro -r ../npt/npt.gro -t ../npt/npt.cpt -p topol.top -o md.tpr

#gmx mdrun -ntomp 20 -deffnm md -plumed plumed.dat
mpiexec -np "$NP" gmx mdrun -gpu_id 0123 -ntomp "$NTHR" -deffnm md -plumed plumed.dat -maxh 36
