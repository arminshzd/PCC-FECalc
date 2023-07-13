#!/bin/sh
##SBATCH --job-name=acpype

# output file (including stderr)
#SBATCH --output=R_%x_%j.out

# email on start, end, and abortion
##SBATCH --mail-type=ALL
##SBATCH --mail-user=arminsh@uchicago.edu

# name of partition to queue on
#SBATCH --account=pi-andrewferguson
#SBATCH --partition=andrewferguson

# max wall time for job (HH:MM:SS)
#SBATCH --time=100:00:00

# number of nodes for this job
#SBATCH --nodes=1

# number of processes to run per node
#SBATCH --ntasks-per-node=8

# reserve the specified node(s) for this job
##SBATCH --exclusive

module load python/anaconda-2022.05
source activate /project/andrewferguson/armin/envs/acpype

acpype -i $1.pdb -b PCC -c bcc -n $2 -a gaff2