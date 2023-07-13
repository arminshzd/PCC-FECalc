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

## Create box
gmx editconf -f PCC_GMX.gro -o PCC_box.gro -c -d 1.0 -bt cubic
## Solvate
gmx solvate -cp PCC_box.gro -cs spc216.gro -o PCC_sol.gro -p topol.top
## Neutralize
CHARGE=$1
echo "System total charge: $CHARGE"
if [ $CHARGE -ne 0 ]
then
    gmx grompp -f ions.mdp -c PCC_sol.gro -p topol.top -o ions.tpr -maxwarn 2
    gmx genion -s ions.tpr -o PCC_sol_ions.gro -p topol.top -pname NA -nname CL -neutral << EOF
4
EOF
    gmx grompp -f em.mdp -c PCC_sol_ions.gro -p topol.top -o em.tpr
else
    gmx grompp -f em.mdp -c PCC_sol.gro -p topol.top -o em.tpr
fi
mpiexec -np 1 mdrun_mpi -ntomp 6 -v -deffnm em

