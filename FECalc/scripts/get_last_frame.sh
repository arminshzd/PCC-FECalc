#! /usr/bin/bash

while getopts f:s:o: flag
do
    case "${flag}" in
        f) trajfile=${OPTARG};; # trajectory file .trr/.xtc
        s) trajset=${OPTARG};; # md settings .tpr
        o) output=${OPTARG};; # output file
    esac
done

module unload openmpi gcc cuda python
#module load openmpi/4.1.1 gcc/7.4.0 cuda/11.2
module load openmpi/4.1.1+gcc-10.1.0 cuda/11.2

#source /project/andrewferguson/armin/grom_new/gromacs-2021.6/installed-files-nompi/bin/GMXRC
source /project/andrewferguson/armin/grom_new/gromacs-2021.6/installed-files-mw2-256/bin/GMXRC

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