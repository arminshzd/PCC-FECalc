#!/bin/sh

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# name of partition to queue on
##SBATCH --account=pi-andrewferguson
##SBATCH --partition=andrewferguson
#SBATCH --partition=gm4-pmext
#SBATCH --qos=gm4-cpu

# max wall time for job (HH:MM:SS)
#SBATCH --time=1-12:00:00

# number of nodes for this job
#SBATCH --nodes=1

# number of processes to run per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

module load python/anaconda-2021.05
source activate /scratch/midway2/arminsh/envs/acpype

acpype -i $1.pdb -b $2 -c bcc -n $3 -a gaff2
