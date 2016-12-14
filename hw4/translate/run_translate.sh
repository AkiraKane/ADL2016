#!/bin/bash

# EXAMPLE USE :
# bash run_translate.sh [testing data] [answer file]
wget -O ./size350_layer2/translate.ckpt-19000 https://www.dropbox.com/s/czgq5z4b9lw8bwy/translate.ckpt-19000?dl=0
wget -O ./size350_layer2/translate.ckpt-19000.meta https://www.dropbox.com/s/uuxt92log78zu0n/translate.ckpt-19000.meta?dl=0

# Original command example
# python translate.py --decode --data_dir translation_data --train_dir size350_layer2 --size=350 --num_layers=2 --test_file test.en --answer_file size350_layer2.txt

python translate.py --decode --data_dir translation_data --train_dir size350_layer2 --size=350 --num_layers=2 --test_file $1 --answer_file $2
