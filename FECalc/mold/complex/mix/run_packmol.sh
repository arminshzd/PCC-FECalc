#!/bin/bash
module unload gcc
module load gcc/10.2.0
/project/andrewferguson/armin/packmol/packmol_mw2/packmol-20.14.2/packmol < ./mix.inp
module unload gcc