#!/bin/sh
##SBATCH --job-name=PCC

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# email on start, end, and abortion
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arminsh@uchicago.edu

# name of partition to queue on
#SBATCH --account=pi-andrewferguson
#SBATCH --partition=andrewferguson-gpu
##SBATCH --partition=gpu

# number of GPU(s) per node, if available
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20

# max wall time for job (HH:MM:SS)
#SBATCH --time=100:00:00

# number of nodes for this job
#SBATCH --nodes=1

# number of processes to run per node
#SBATCH --ntasks-per-node=1

# reserve the specified node(s) for this job
##SBATCH --exclusive

module load python/anaconda-2022.05  openmpi/4.1.1 gcc/10.2.0 cuda/11.2 fftw3/3.3.9 gsl/2.7 lapack/3.10.0

gmx grompp -f md.mdp -c ../npt/npt.gro -r ../npt/npt.gro -t ../npt/npt.cpt -p topol.top -o md.tpr

mpiexec -np 4 --oversubscribe mdrun_mpi -ntomp 5 -v -deffnm md -plumed plumed.dat
