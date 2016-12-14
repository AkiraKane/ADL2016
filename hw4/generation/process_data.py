'''

    [
        "?confirm(goodformeal=dont_care)",
        "can i confirm you do not care what type of meal it is",
        "let me confirm you do not care about meal"
    ],



'''
# -*- coding: utf-8 -*-
import json
import os
import sys

def output_txt_file(data_file, query_file_name, reply_file_name):
    query_file = open(query_file_name,"w")
    reply_file = open(reply_file_name,"w")
    slot_list = []
    sentence_list = []
    with open(data_file) as json_data:
        data = json.load(json_data)
        for section in data:
        # Case 1 : goodbye section
            if "goodbye" in section[0]:
                query_file.write(section[0])
                query_file.write("\n")
                query_file.write(section[0])
                query_file.write("\n")
                reply_file.write(section[1])
                reply_file.write("\n")
                reply_file.write(section[2])
                reply_file.write("\n")
                continue
        # Case 2 : request section
            if "?request" in section[0]:
                query_file.write(section[0])
                query_file.write("\n")
                query_file.write(section[0])
                query_file.write("\n")
                reply_file.write(section[1])
                reply_file.write("\n")
                reply_file.write(section[2])
                reply_file.write("\n")
                continue

        # Case 2 : reqmore section
            if "?reqmore" in section[0]:
                query_file.write(section[0])
                query_file.write("\n")
                query_file.write(section[0])
                query_file.write("\n")
                reply_file.write(section[1])
                reply_file.write("\n")
                reply_file.write(section[2])
                reply_file.write("\n")
                continue
        # All the other cases
            try:    
                ''' 
                query =  section[0]
                query_type = query.split("(")[0]
                slot_section = query.split("(")[1].split(")")[0] 
                local_list = []
                for slot in slot_section.split(";"):
                    slot_type = slot.split("=")[0]
                    slot_value = slot.split("=")[1]
                    slot_value = slot_value.replace("'","")
                    #print slot_value
                    local_list.append((slot_type,slot_value))
                reply_1 = section[1].strip()
                for item in local_list:
                    if item[1] in reply_1:
                        reply_1 = reply_1.replace(item[1], "__"+item[0]+"__")
                reply_2 = section[2]    
                for item in local_list:
                    if item[1] in reply_2:
                        reply_2 = reply_2.replace(item[1], "__"+item[0]+"__")
                slot_type_list = [ ("__" +item[0] + "__") for item in local_list ]
                query_sentence = query_type + " " + " ".join( slot_type_list ) 

                query_file.write(query_sentence)
                query_file.write("\n")
                query_file.write(query_sentence)
                query_file.write("\n")
                reply_file.write(reply_1)
                reply_file.write("\n")
                reply_file.write(reply_2)
                reply_file.write("\n")
                '''
                '''
    [
        "inform(name='alamo square seafood grill';area='friendship village';pricerange=moderate)",
        "alamo square seafood grill is a nice restaurant in the area of friendship village and it is in the moderate price range",
        "alamo square seafood grill is a nice place , it is in the area of friendship village and it is in the moderate price range"
    ],
                '''
                query = section[0]
                query_type = query.split("(")[0]
                slot_section = query.split("(")[1].split(")")[0] 
                reply_1 = section[1]
                reply_2 = section[2]
                local_list = []
                for slot in slot_section.split(";"):
                    slot_type = slot.split("=")[0]
                    slot_value = slot.split("=")[1]
                    slot_value = slot_value.replace("'","")
                    #print slot_value
                    local_list.append((slot_type,slot_value))
                query_sentence = query_type 
                for item in local_list:
                    query_sentence = query_sentence + " "
                    query_sentence = query_sentence + item[0]
                    query_sentence = query_sentence + " "
                    query_sentence = query_sentence + item[1]
                query_file.write(query_sentence)
                query_file.write("\n")
                query_file.write(query_sentence)
                query_file.write("\n")
                reply_file.write(reply_1)
                reply_file.write("\n")
                reply_file.write(reply_2)
                reply_file.write("\n")
            except:
                pass

    query_file.close()
    reply_file.close()
 
data_path = "./NLG_data"
file_name = "train.json"
train_data_file = os.path.join(data_path,file_name)

train_query_file_name = "train.que"
train_reply_file_name = "train.rep"

# Output train file
#output_txt_file(train_data_file, train_query_file_name, train_reply_file_name)  ####

data_path = "./NLG_data"
dev_file_name = "valid.json"
dev_data_file = os.path.join(data_path,dev_file_name)

dev_query_file_name = "dev.que"
dev_reply_file_name = "dev.rep"
# Output dev file
output_txt_file(dev_data_file, dev_query_file_name, dev_reply_file_name)  ####



