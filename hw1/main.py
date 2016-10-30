#encoding=utf8

# reference to https://github.com/GradySimon/tensorflow-glove/blob/master/tf_glove.py
'''
Credits
Naturally, most of the credit goes to Jeffery Pennington, Richard Socher, and Christopher Manning, who developed the model, published a paper about it, and released an implementation in C.

Thanks also to Jon Gauthier (@hans), who wrote a Python implementation of the model and a blog post describing that implementation, which were both very useful references as well.
'''

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


embedding_size_p = 200
context_size_p = 10
max_vocab_size_p = 100000000000000000000
min_occurences_p = 5
epochs = 150



model = tf_glove.GloVeModel(embedding_size=embedding_size_p, context_size=context_size_p, max_vocab_size=max_vocab_size_p, min_occurrences=min_occurences_p)

# text8 preprocessing 
data = open(train_data,"r").read()
data = data.split(" ")
corpus = []
corpus.append(data)
corpus_set = set(data)

model.fit_to_corpus(corpus)
model.train(num_epochs=epochs)

# save the answer 
embedding_file_txt = "glove_result.txt"

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
