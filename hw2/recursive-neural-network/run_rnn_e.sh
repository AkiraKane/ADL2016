#!/bin/bash

# testing data file :$1  # no need for this file
# testing tree file :$2
# output file : $3

# TO DO : 
# Download the embeddings and the model weight
#rnn_embed=200_l2=0.020000_lr=0.010000_epoch=50.weights

#RVNN_EMBEDDING.txt
#https://drive.google.com/file/d/0B1kSNOCNNMsbV204T0VZNVlHY2s/view?usp=sharing
#wget --no-check-certificate --ignore-length "https://drive.google.com/uc?export=download&id=0B1kSNOCNNMsbV204T0VZNVlHY2s" -O RVNN_EMBEDDING.txt
#wget "https://googledrive.com/host/0B1kSNOCNNMsbV204T0VZNVlHY2s" -o RVNN_EMBEDDING.txt

# weight file
# https://drive.google.com/file/d/0B1kSNOCNNMsbSHZ3dU1jYUtyZmM/view?usp=sharing
#wget --no-check-certificate --ignore-length "https://drive.google.com/uc?export=download&id=0B1kSNOCNNMsbSHZ3dU1jYUtyZmM" -O rnn_embed=200_l2=0.020000_lr=0.010000_epoch=50.weights


# preprocess the tree_file 
python tree_preprocess.py --treefile $2

python rnn.py --revised_tree revised_tree_file.txt --outputfile_name=$3

# wight file :
#/weights/weighting_file
#rnn_embed=200_l2=0.020000_lr=0.010000_epoch=50.weights

