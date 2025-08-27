#!/bin/sh

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# name of partition to queue on
#SBATCH --partition=
#SBATCH --qos=

# max wall time for job (HH:MM:SS)
#SBATCH --time=1-12:00:00

# number of GPU(s) per node, if available
#SBATCH --gres=gpu:1

# number of nodes for this job
#SBATCH --nodes=1

# number of processes to run per node
#SBATCH --ntasks-per-node=4

# number of threads per cpu
#SBATCH --cpus-per-task=5

NCPU=$(($SLURM_NTASKS_PER_NODE))
NTHR=$(($SLURM_CPUS_PER_TASK))
NNOD=$(($SLURM_JOB_NUM_NODES))

NP=$(($NCPU * $NNOD * $NTHR))

module unload openmpi gcc cuda python
module load openmpi/4.1.1+gcc-10.1.0 cuda/11.2

source /path/to/gromacs/bin/GMXRC

## Create box
gmx editconf -f complex.pdb -o complex_box.gro -c -bt cubic -box "$1"
## Solvate
gmx solvate -cp complex_box.gro -cs spc216.gro -o complex_sol.gro -p topol.top
## Neutralize
CHARGE=$2
echo "System total charge: $CHARGE"
if [ $CHARGE -ne 0 ]
then
    gmx grompp -f ions.mdp -c complex_sol.gro -p topol.top -o ions.tpr -maxwarn 2
    gmx genion -s ions.tpr -o complex_sol_ions.gro -p topol.top -pname NA -nname CL -neutral << EOF
5
EOF
    gmx grompp -f em.mdp -c complex_sol_ions.gro -p topol.top -o em.tpr -maxwarn 1
else
    gmx grompp -f em.mdp -c complex_sol.gro -p topol.top -o em.tpr -maxwarn 1
fi

gmx mdrun -ntomp "$NP" -deffnm em
