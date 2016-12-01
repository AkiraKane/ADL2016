#!/usr/bin/python
from random import shuffle
import argparse
import os, sys


'''
BOS i want to fly from boston at 838 am and arrive in denver at 1110 in the morning EOS O O O O O O B-fromloc.city_name O B-depart_time.time I-depart_time.time O O O B-toloc.city_name O B-arrive_time.time O O B-arrive_time.period_of_day atis_flight

# train.seq.in
"what's the lowest round trip fare from dallas to atlanta"
# train.seq.out
"O O B-cost_relative B-round_trip I-round_trip O O B-fromloc.city_name O B-toloc.city_name"
# train.label
airfare
'''

data_dir = "./Slot_Filling_Data" 
file_name = "atis.train.w-intent.iob"
data_file = data_dir + file_name 

# 4978  : 4000, 900, 78
train_num = 4000
valid_num = 4500
# test_num is the rest data

def product_data_list(data_file):
    train_lines = open(data_file,"r").readlines()
    data_list = []
    for line in train_lines:
        line = line.strip()
        label = line.split()[-1]
        front_line = " ".join(line.split()[:-1])
        seq_in = front_line.split("EOS")[0].strip()
        seq_out = front_line.split("EOS")[1].strip()
        data_list.append((seq_in,seq_out,label))    
    return data_list



def create_dir_and_write_new_data(data_list, data_dir, mode="train",train_num=train_num,valid_num=train_num):
    if mode == "train":
        data_list = data_list[:train_num]
    if mode == "valid":
        data_list = data_list[train_num:valid_num]
    if mode == "test":
        data_list = data_list[valid_num:]
   
    data_path = os.path.join(data_dir,mode)

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    seq_in = mode + ".seq.in"
    seq_out = mode + ".seq.out"
    label = mode + ".label"
    seq_in_name = os.path.join(data_path,seq_in)
    seq_out_name = os.path.join(data_path,seq_out)
    label_name = os.path.join(data_path,label)
    seq_in_file = open(seq_in_name,"w")
    seq_out_file = open(seq_out_name,"w")
    label_file = open(label_name,"w")
    for _tuple in data_list:
        (_in, _out ,_label) = _tuple
        seq_in_file.write(_in) 
        seq_in_file.write("\n") 
        seq_out_file.write(_out) 
        seq_out_file.write("\n") 
        label_file.write(_label) 
        label_file.write("\n") 

    seq_in_file.close()     
    seq_out_file.close()     
    label_file.close()     

if __name__ == '__main__':
    '''
    data_dir = "./Slot_Filling_Data/" 
    file_name = "atis.train.w-intent.iob"
    data_file = data_dir + file_name 

    # 4978  : 4000, 900, 78
    train_num = 4000
    valid_num = 4500
    # test_num is the rest data

    data_list = product_data_list(data_file)

    shuffle(data_list)

    create_dir_and_write_new_data(data_list, data_dir, mode="train",train_num=train_num,valid_num=valid_num)
    create_dir_and_write_new_data(data_list, data_dir, mode="valid",train_num=train_num,valid_num=valid_num)
    create_dir_and_write_new_data(data_list, data_dir, mode="test",train_num=train_num,valid_num=valid_num)
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_file",type=str)
    args = parser.parse_args()
    test_file_name = args.test_file

    # Parse the test file to "BOS + Original sent"
    # 700
    #test_lines = open("./Slot_Filling_Data_2/atis.test.iob","r").readlines()
    #test_lines = open("./Intent_Prediction_Data/atis.test.iob","r").readlines()
    test_lines = open(test_file_name,"r").readlines()
    test_file_final = open("Slot_Filling_test.txt","w")
    for line in test_lines:
        line = line.split("EOS")[0].strip()
        test_file_final.write(str(line))
        test_file_final.write("\n")
    test_file_final.close()
