#!/bin/sh

module load python/anaconda-2021.05
source activate /project/andrewferguson/armin/envs/ase

python pcc_submit.py -p $1 -t $2 >> "$1"_"$2".out 2>&1 &