#!/bin/sh
mkdir $2
mkdir temp


# run command for wordvec.py
# although there is a eval data input, the code isn't doing the evaluation
python word2vec_optimized.py --train_data=$1 --eval_data=/word2vec/questions-words.txt --save_path=/temp/

python /filterVocab/filterVocab.py /filterVocab/fullVocab.txt < /temp/wordtovec_result.txt > $2/filter_word2vec.txt

# run command for glove.py
python main.py --train_data=$1 --save_path=/temp/

python /filterVocab/filterVocab.py /filterVocab/fullVocab.txt < /temp/glove_result.txt > $2/filter_glove.txt

