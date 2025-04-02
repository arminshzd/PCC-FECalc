#!/bin/sh

module load python/anaconda-2021.05
source activate /scratch/midway2/arminsh/envs/ase

python sub_PCCBuilder.py -p $1 >> "$1".out 2>&1 &
