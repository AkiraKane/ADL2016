#encoding=utf8
import tensorflow as tf
import tf_glove
import os
import argparse


flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model.")
flags.DEFINE_string(
    "train_data", None,
    "Training data. E.g., unzipped file http://mattmahoney.net/dc/text8.zip.")

FLAGS = flags.FLAGS
# The training text file.
train_data = FLAGS.train_data

# Where to write out summaries.
save_path = FLAGS.save_path

# The text file for eval.
#eval_data = FLAGS.eval_data

'''
# Use nargs to specify how many arguments an option should take.
ap = argparse.ArgumentParser()
ap.add_argument('-train_data', nargs=1, required=True)
ap.add_argument('-save_path', nargs=1, required=True)

# An illustration of how access the arguments.
opts = ap.parse_args()

datafile = opts.train_data
save_path = opts.save_path
'''


embedding_size_p = 50
context_size_p = 5
max_vocab_size_p = 1000
min_occurences_p = 15
epochs = 1



model = tf_glove.GloVeModel(embedding_size=embedding_size_p, context_size=context_size_p, max_vocab_size=max_vocab_size_p, min_occurrences=min_occurences_p)

# data preprocessing : using text8
#datafile = "/home/b01901073/adl/hw1/tensorflow/tensorflow/models/embedding/text8"

# text8 preprocessing 
data = open(train_data,"r").read()
data = data.split(" ")
corpus = []
corpus.append(data)
corpus_set = set(data)

model.fit_to_corpus(corpus)
model.train(num_epochs=epochs)

# save the answer 
#embedding_path = "/home/b01901073/adl/hw1/tensorflow/tensorflow/models/embedding/test/glove_txt/"
embedding_file_txt = "glove_test_file_final.txt"

vocab_size = model.vocab_size

with open(os.path.join(save_path, embedding_file_txt), "w") as f:
    for word in corpus_set:
        try:
            row = model.embedding_for(word).tolist()
            f.write("%s" % (word))
            f.write(" ")
            for j in range(len(row)):
                f.write(str(row[j]))
                f.write(" ")
            f.write("\n")
        except:
                pass

#model.generate_tsne()
