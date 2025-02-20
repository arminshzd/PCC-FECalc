#!/bin/sh

module load python/anaconda-2021.05
source activate /scratch/midway2/arminsh/envs/ase

python sub_PCCPCC.py -p $1 -t $2 >> "$1"_"$2".out 2>&1 &
