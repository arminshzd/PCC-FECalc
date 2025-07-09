#!/bin/sh

module load python/anaconda-2021.05
source activate /scratch/midway2/arminsh/envs/ase

python pcc_submit.py -p $1 -s $2 >> "$1".log 2>&1 &
