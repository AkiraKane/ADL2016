#!/bin/bash

# EXAMPLE USE :
# bash run_generation.sh [testing data] [answer file]

# Download tensorflow model from dropbox
# wget -O /var/cache/foobar/stackexchange-site-list.txt http://url.to/stackexchange-site-list.txt
wget -O ./size300_layer2_version1_learn0.3/translate.ckpt-5650 https://www.dropbox.com/s/ibjmyc7lv8s5cap/translate.ckpt-5650?dl=0
wget -O ./size300_layer2_version1_learn0.3/translate.ckpt-5650.meta https://www.dropbox.com/s/qsp08vd2a1apxxi/translate.ckpt-5650.meta?dl=0

# original command sample
#python generation.py --decode --data_dir generation_data --train_dir size300_layer2_version1_learn0.3 --size=300 --num_layers=2 --test_file test.txt --answer_file size300_layer2_version1_learn0.3_simplified.txt

python generation.py --decode --data_dir generation_data --train_dir size300_layer2_version1_learn0.3 --size=300 --num_layers=2 --test_file $1 --answer_file $2
