#!/bin/bash

solution=$1
filename=$(basename $solution)
submissionName=${filename%.pyz}

path="."
mkdir $path/evalDirectory
eval=$path/evalDirectory

testImagesDirectory="evaluation/data/LPCVC_Val/LPCVC_Val/IMG/val"

python3 $solution -i ${testImagesDirectory} -o $eval

rm -rf $eval
