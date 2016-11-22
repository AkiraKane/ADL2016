import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import itertools
import random
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab

RESET_AFTER = 50

flags = tf.app.flags

flags.DEFINE_string("revised_tree", None,"Testing data file")
flags.DEFINE_string("outputfile_name", None, "the predictions for test file ")

FLAGS = flags.FLAGS

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation.
    """
    embed_size = 200 # use self-trained word2vec vectors
    label_size = 2  # sentiment is 0 or 1, the softmax prediction size
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 50
    lr = 0.01
    l2 = 0.02
    model_name = 'rnn_embed=%d_l2=%f_lr=%f_epoch=%d.weights'%(embed_size, l2, lr, max_epochs)


class RNN_Model():

    def load_data(self,LOAD_DATA=False):
        """Loads train/dev/test data and builds vocabulary."""
        if LOAD_DATA:
            self.vocab = Vocab() # only initialize the Vocab class because of the embedding matrix
        else:
            self.train_data, self.dev_data, self.test_data = tr.simplified_data(600,40)
        #self.train_data, self.dev_data , self.test_data = tr.simplified_data(2000, 500)
            # build vocab from training data
            self.vocab = Vocab()
            train_sents = [t.get_words() for t in self.train_data]
            self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    def inference(self, tree, predict_only_root=False):
        """For a given tree build the RNN models computation graph up to where it
            may be used for inference.
        Args:
            tree: a Tree object on which to build the computation graph for the RNN
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        node_tensors = self.add_model(tree.root)
        if predict_only_root:
            node_tensors = node_tensors[tree.root]
        else:
            node_tensors = [tensor for node, tensor in node_tensors.iteritems() if node.label!=2]
            node_tensors = tf.concat(0, node_tensors)
        return self.add_projections(node_tensors)

    def add_model_vars(self):
        '''
        You model contains the following parameters:
            embedding:  tensor(vocab_size, embed_size)
            W1:         tensor(2* embed_size, embed_size)
            b1:         tensor(1, embed_size)
            U:          tensor(embed_size, output_size)
            bs:         tensor(1, output_size)
        Hint: Add the tensorflow variables to the graph here and *reuse* them while building
                the compution graphs for composition and projection for each tree
        Hint: Use a variable_scope "Composition" for the composition layer, and
              "Projection") for the linear transformations preceding the softmax.
        '''
        #self.emb_words = emb_words
        #self.emb_numpymatrix = word2vec_embedding
        with tf.variable_scope('Composition'):
            ### YOUR CODE HERE
            # USE THE PRETRAINED WORD ENBEDDING IN SELF.VOCAB
            word2vec_embedding = self.vocab.emb_numpymatrix 
            E = tf.constant(word2vec_embedding)
            E = tf.cast(E, tf.float32)
            tf.get_variable("Word2vec_E", initializer = E,  trainable=False)
            #tf.get_variable("Word2vec_E", initializer = word2vec_embedding,  trainable=False)
            #tf.get_variable('embedding', shape=[self.vocab.total_words, self.config.embed_size])
            tf.get_variable('W1', shape=[2*self.config.embed_size, self.config.embed_size])
            tf.get_variable('b1',shape=[1,self.config.embed_size])

            
            ### END YOUR CODE
        with tf.variable_scope('Projection'):
            ### YOUR CODE HERE
            tf.get_variable('U',shape=[self.config.embed_size,self.config.label_size])
            tf.get_variable('bs',shape=[1, self.config.label_size])
            ### END YOUR CODE

    def add_model(self, node):
        """Recursively build the model to compute the phrase embeddings in the tree

        Hint: Refer to tree.py and vocab.py before you start. Refer to
              the model's vocab with self.vocab
        Hint: Reuse the "Composition" variable_scope here
        Hint: Store a node's vector representation in node.tensor so it can be
              used by it's parent
        Hint: If node is a leaf node, it's vector representation is just that of the
              word vector (see tf.gather()).
        Args:
            node: a Node object
        Returns:
            node_tensors: Dict: key = Node, value = tensor(1, embed_size)
        """
        with tf.variable_scope('Composition', reuse=True):
            ### YOUR CODE HERE
            #embedding = tf.get_variable('embedding')
            embedding = tf.get_variable('Word2vec_E')
            W1=tf.get_variable('W1')
            b1=tf.get_variable('b1')
            ### END YOUR CODE


        node_tensors = dict()
        curr_node_tensor = None
        if node.isLeaf:
            ### YOUR CODE HERE
            #idx = self.vocab.encode(node.word)
            try:
                idx = self.vocab.emb_wordtoindex(node.word)
            except:
                idx = self.vocab.emb_wordtoindex("UNK")
            h = tf.gather(embedding, indices=idx)
            curr_node_tensor = tf.expand_dims(h, 0)
            ### END YOUR CODE
        else:
            node_tensors.update(self.add_model(node.left))
            node_tensors.update(self.add_model(node.right))
            ### YOUR CODE HERE
            HlHr=tf.concat(1, [node_tensors[node.left], node_tensors[node.right]])
            curr_node_tensor = tf.nn.relu(tf.matmul(HlHr, W1) + b1)
            ### END YOUR CODE
        node_tensors[node] = curr_node_tensor
        return node_tensors

    def add_projections(self, node_tensors):
        """Add projections to the composition vectors to compute the raw sentiment scores

        Hint: Reuse the "Projection" variable_scope here
        Args:
            node_tensors: tensor(?, embed_size)
        Returns:
            output: tensor(?, label_size)
        """
        logits = None
        ### YOUR CODE HERE
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U')
            bs = tf.get_variable('bs')
            logits = tf.matmul(node_tensors, U) + bs
        ### END YOUR CODE
        return logits

    def loss(self, logits, labels):
        """Adds loss ops to the computational graph.

        Hint: Use sparse_softmax_cross_entropy_with_logits
        Hint: Remember to add l2_loss (see tf.nn.l2_loss)
        Args:
            logits: tensor(num_nodes, output_size)
            labels: python list, len = num_nodes
        Returns:
            loss: tensor 0-D
        """
        loss = None
        # YOUR CODE HERE
        with tf.variable_scope('Composition', reuse=True):
            W1 = tf.get_variable('W1')
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U')
        
        l2loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(U)

        cross_entropy = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
        loss = cross_entropy + self.config.l2 * l2loss

        # sparse_softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)
        # tf.add_to_collection('total_loss', tf.reduce_sum(sparse_softmax))
        # for variable in [W1, U]:
        #     tf.add_to_collection('total_loss', self.config.l2 * tf.nn.l2_loss(variable))
        # loss = tf.add_n(tf.get_collection('total_loss'))
        # END YOUR CODE
        return loss

    def training(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.GradientDescentOptimizer for this model.
                Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: tensor 0-D
        Returns:
            train_op: tensorflow op for training.
        """
        train_op = None
        # YOUR CODE HERE
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        # END YOUR CODE
        return train_op

    def predictions(self, y):
        """Returns predictions from sparse scores

        Args:
            y: tensor(?, label_size)
        Returns:
            predictions: tensor(?,1)
        """
        predictions = None
        # YOUR CODE HERE
        predictions = tf.argmax(y, 1)
        # END YOUR CODE
        return predictions

    def __init__(self, config, LOAD_DATA=True):
        self.config = config 
        self.load_data(LOAD_DATA=True)

    def predict(self, trees, weights_path, get_loss = False):
        """Make predictions from the provided model."""
        results = []
        losses = []
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        config = tf.ConfigProto(gpu_options=gpu_options)
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True
        for i in xrange(int(math.ceil(len(trees)/float(RESET_AFTER)))):
            with tf.Graph().as_default(), tf.Session(config=config) as sess:  ###########################
                self.add_model_vars() ##############################
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i*RESET_AFTER: (i+1)*RESET_AFTER]:
                    if tree=="empty":  # for the testing tree that has multiple childs
                        root_prediction = int(random.getrandbits(1)) # randomly predict the answer (choose evenly from 0 or 1)
                        results.append(root_prediction)
                        continue
  
                    logits = self.inference(tree, True) ##############################
                    predictions = self.predictions(logits)
                    root_prediction = sess.run(predictions)[0]
                    if get_loss:
                        #root_label = tree.root.label
                        root_label = tree.label
                        loss = sess.run(self.loss(logits, [root_label]))   ###############################
                        losses.append(loss)
                    results.append(root_prediction)
        return results, losses

    def run_epoch(self, new_model = False, verbose=True):
        step = 0
        loss_history = []
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        config = tf.ConfigProto(gpu_options=gpu_options)
        #config = tf.ConfigProto()
        #config.gpu_options.allow_growth=True
        while step < len(self.train_data):
            with tf.Graph().as_default(), tf.Session(config=config) as sess:
                self.add_model_vars()
                if new_model:
                    init = tf.initialize_all_variables()
                    sess.run(init)
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess, './weights/%s.temp'%self.config.model_name)
                for _ in xrange(RESET_AFTER):
                    if step>=len(self.train_data):
                        break
                    tree = self.train_data[step]
                    #logits = self.inference(tree)
                    #labels = [l for l in tree.labels if l!=2]
                    logits = self.inference(tree,True)
                    label = tree.label
                    loss = self.loss(logits, [label])
                    train_op = self.training(loss)
                    loss, _ = sess.run([loss, train_op])
                    loss_history.append(loss)
                    if verbose:
                        sys.stdout.write('\r{} / {} :    loss = {}'.format(
                            step, len(self.train_data), np.mean(loss_history)))
                        sys.stdout.flush()
                    step+=1
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(sess, './weights/%s.temp'%self.config.model_name)
        train_preds, _ = self.predict(self.train_data, './weights/%s.temp'%self.config.model_name)
        val_preds, val_losses = self.predict(self.dev_data, './weights/%s.temp'%self.config.model_name, get_loss=True)
        #val_preds, _ = self.predict(self.dev_data, './weights/%s.temp'%self.config.model_name, get_loss=True)
        #train_labels = [t.root.label for t in self.train_data]
        #val_labels = [t.root.label for t in self.dev_data]
        train_labels = [t.label for t in self.train_data]
        val_labels = [t.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()

        print
        print 'Training acc (only root node): {}'.format(train_acc)
        print 'Valiation acc (only root node): {}'.format(val_acc)
        print self.make_conf(train_labels, train_preds)
        print self.make_conf(val_labels, val_preds)
        return train_acc, val_acc, loss_history, np.mean(val_losses)
        #return train_acc, val_acc, loss_history

    def train(self, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in xrange(self.config.max_epochs):
            print 'epoch %d'%epoch
            if epoch==0:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(new_model=True)
                #train_acc, val_acc, loss_history = self.run_epoch(new_model=True)
            else:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch()
                #train_acc, val_acc, loss_history = self.run_epoch()
            complete_loss_history.extend(loss_history)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            #lr annealing
            epoch_loss = np.mean(loss_history)
            if epoch_loss>prev_epoch_loss*self.config.anneal_threshold:
                self.config.lr/=self.config.anneal_by
                print 'annealed lr to %f'%self.config.lr
            prev_epoch_loss = epoch_loss


            #save if model has improved on val_loss
            if val_loss < best_val_loss:
                 shutil.copyfile('./weights/%s.temp'%self.config.model_name, './weights/%s'%self.config.model_name)
                 best_val_loss = val_loss
                 best_val_epoch = epoch

            # if model has not imprvoved for a while stop
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch
                #break

        if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

        #print '\n\nstopped at %d\n'%stopped
        return {
            'loss_history': complete_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            }

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        for l,p in itertools.izip(labels, predictions):
            confmat[l, p] += 1
        return confmat


def test_RNN():
    """Test RVNN model implementation. 
    """
    
    ### Training process ###
    '''
    config = Config()
    model = RNN_Model(config)
    start_time = time.time()
    stats = model.train(verbose=True)
    print 'Training time: {}'.format(time.time() - start_time)

    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    loss_filename = config.model_name + "loss_history" + ".png"
    plt.savefig(loss_filename)

    print 'Test'
    print '=-=-='
    '''
    ### Training process End ###

   

    ### Testing process ###
    #predictions, _ = model.predict(model.test_data, './weights/%s'%model.config.model_name)
    config = Config()
    model = RNN_Model(config, LOAD_DATA=False)

    #test_trees = tr.loadTrees('testing_data.txt.treerevised.txt', predicting_test=True)
    test_trees = tr.loadTrees(FLAGS.revised_tree , predicting_test=True)

    predictions, _ = model.predict(test_trees, './weights/rnn_embed=200_l2=0.020000_lr=0.010000_epoch=50.weights')

    f = open(FLAGS.outputfile_name , "w")
    for prediction in predictions:
        f.write(str(int(prediction)))    
        f.write("\n")    
    ### Testing process End ###
    
    #labels = [t.root.label for t in model.test_data]
    #labels = [t.label for t in model.test_data]
    #test_acc = np.equal(predictions, labels).mean()
    #print 'Test acc: {}'.format(test_acc)

if __name__ == "__main__":
    test_RNN()
