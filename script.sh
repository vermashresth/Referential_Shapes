#!/bin/bash
#Set job requirements
#SBATCH -t 24:00:00
#SBATCH -p gpu

#Loading modules
#module load python

#Copy input data to scratch and create output directory
#cp -r $HOME/input_dir "$TMPDIR"
#mkdir "$TMPDIR"/output_dir

#Run program
python test.py

#Copy output data from scratch to home
#cp -r "$TMPDIR"/output_dir $HOME
