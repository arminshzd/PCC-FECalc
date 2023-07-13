#! /usr/bin/bash

while getopts f:s:o: flag
do
    case "${flag}" in
        f) trajfile=${OPTARG};; # trajectory file .trr/.xtc
        s) trajset=${OPTARG};; # md settings .tpr
        o) output=${OPTARG};; # output file
    esac
done

module load python/anaconda-2022.05 openmpi/4.1.1 gcc/10.2.0 cuda/11.2 fftw3/3.3.9 gsl/2.7 lapack/3.10.0

parent_dir="$(dirname "$trajfile")"
frames_dir="$parent_dir"/frames.temp
ndx_dir="$parent_dir"/lastframe.ndx


# get number of frames
gmx check -f $trajfile 2>&1 | tee $frames_dir > /dev/null
nframes=$(grep "Step" $frames_dir | awk '{print $NF}')

# make lastframe.temp
cat << EOF > $ndx_dir
[ last_frame ]
$nframes
EOF

# Call trjconv
gmx trjconv -s $trajset -f $trajfile -o $output -pbc whole -conect -fr $ndx_dir <<EOF
2
EOF

rm $frames_dir