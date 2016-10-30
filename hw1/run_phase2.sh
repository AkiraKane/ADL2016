#!/bin/sh
mkdir $2
mkdir temp

# run command for wordvec.py
# although there is a eval data input, the code isn't doing the evaluation
python word2vec_optimized_ptt.py --train_data=$1 --eval_data=word2vec/questions-words.txt --save_path=temp

python filterVocab/filterVocab.py filterVocab/fullVocab_phase2.txt < temp/ptt_result.txt > $2/filter_vec.txt

