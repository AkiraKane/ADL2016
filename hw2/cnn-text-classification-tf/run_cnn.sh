#!/bin/bash

# $1  : inputfile
# $2  : outputfile

python3 eval.py --inputfile $1 --outputfile $2 --checkpoint_dir="./model_save/checkpoints"
