#!/bin/bash
module unload gcc
module load gcc/10.2.0
packmol < ./mix.inp
module unload gcc