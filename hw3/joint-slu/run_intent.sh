#!/bin/bash

# EXAMPLE USE :
# bash run_intent.sh [testing data] [answer file]

# produce Slot_Filling_test.txt from testig data 
python process_data.py -t $1 

model_dir=model_final
max_sequence_length=50  # maix length for train/valid/test sequence
use_local_context=False # boolean, whether to use local context
DNN_at_output=True # boolean, set to True to use one hidden layer DNN at task output
data_dir=./Slot_Filling_Data            # train : val : 4000 : 500 : 487
#train_history_file=train4000_rnn_nolocal # when traning remembber to give name for csv


decode_file=True # use when predicting answer
test_file=Slot_Filling_test.txt # use when predicting answer
test_output_file=$2 # # use when predicting answer

python run_rnn_joint_for_intent.py --data_dir $data_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --use_local_context $use_local_context \
      --DNN_at_output $DNN_at_output \
      --decode_file $decode_file  \
      --test_file $test_file \
      --test_output_file $test_output_file 
