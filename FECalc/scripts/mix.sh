#! /usr/bin/bash

while getopts p:f:o: flag
do
    case "${flag}" in
        p) pcc=${OPTARG};;
        f) fen=${OPTARG};;
        o) output=${OPTARG};;
    esac
done

parent_dir="$(dirname "$output")"
mix_dir="$parent_dir"/mix.inp

cat <<EOF > $mix_dir
tolerance 2.0
filetype pdb
output $output

structure $pcc
    number 1
    inside cube 0 0 0 30
end structure

structure $fen
    number 1
    inside cube 0 0 0 30
end structure
EOF

/project/andrewferguson/armin/packmol/packmol-20.14.2/packmol < $mix_dir