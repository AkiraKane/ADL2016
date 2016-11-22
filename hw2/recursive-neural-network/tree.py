import random
import pdb

UNK = 'UNK'
# This file contains the dataset in a useful way. We populate a list of
# Trees to train/test our Neural Nets such that each Tree contains any
# number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..


class Node:  # a node in the tree
    #def __init__(self, label, word=None):
    def __init__(self, word=None):
        #self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)


class Tree:
    def __init__(self, treeString, openChar='(', closeChar=')', fixed=False):
        tokens = []
        # to build a fixed tree because of the weird multiple childs case
        if fixed:
            self.open = '('
            self.close = ')'
            self.label = 0
            return None
        # normal way to build the tree    
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        # get list of labels as obtained through a post-order traversal
        #self.labels = get_labels(self.root)
        #self.num_words = len(self.labels) 

        ##################
        self.label = 0
        ##################

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree" 
        assert tokens[-1] == self.close, "Malformed tree"

        split = 1  # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
               countOpen += 1
            if tokens[split] == self.close:
               countClose += 1
            split += 1

	# New node
        node = Node()  

        node.parent = parent

	# leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[1:-1]).lower()  # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[1:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node)
	###################################

        return node

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words


def leftTraverse(node, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


def get_labels(node):
    if node is None:
        return []
    return get_labels(node.left) + get_labels(node.right) + [node.label]


def clearFprop(node, words):
    node.fprop = False


#def loadTrees(dataSet='train'):
def loadTrees(dataSet='training_data.pos.treerevised.txt', predicting_test=False):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    #file = 'trees/%s.txt' % dataSet
    #print "Loading %s trees.." % dataSet
    file = dataSet
    print "Loading %s trees.." % dataSet
    trees = []
    if predicting_test:
        with open(file, 'r') as fid:
            #trees = [Tree(l) for l in fid.readlines()]
            for l in fid.readlines():
                try:
                    tree = Tree(l)
                    trees.append(tree)
                except:
                    trees.append("empty")
    else:
        with open(file, 'r') as fid:
            #trees = [Tree(l) for l in fid.readlines()]
            for l in fid.readlines():
                try:
                    tree = Tree(l)
                    trees.append(tree)
                except:     
                    #word_list = preprocess_multiplechild_tree(l)
                    #final_merged_node = build_fixedtree(word_list)
                    #tree = Tree("",fixed=True)
                    #tree.root = final_merged_node
                    #trees.append(tree) 
                    pass
    return trees

def preprocess_multiplechild_tree(treestring):
    treestring = "".join(treestring.split())
    word_list = treestring.split("(")  
    word_list = [word for word in word_list if word != ""]
    word_list = [word.replace(")","") for word in word_list if word != ""]
    return word_list

'''
class Node:  # a node in the tree
    #def __init__(self, label, word=None):
    def __init__(self, word=None):
        #self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)
'''


def build_fixedtree(word_list):
    sent_len = len(word_list)
    # build first merged_subtree
    node_1 = Node(word_list[-1])
    node_1.isLeaf = True
    node_2 = Node(word_list[-2])
    node_2.isLeaf = True 
    node_1and2 = Node()
    node_1and2.left = node_2
    node_1and2.right = node_1
    node_1.parent = node_1and2
    node_2.parent = node_1and2
    sub_tree = node_1and2
    #shift the pointer from right to left
    for point_index in range(-3,-(sent_len),-1):
        new_word = word_list[point_index]
        sub_tree = build_block(new_word, sub_tree)
    return sub_tree    

def build_block(new_word, sub_tree):
    new_node = Node(new_word)
    new_node.isLeaf = True
    merged_tree = Node()
    new_node.parent = merged_tree
    sub_tree.parent = merged_tree
    merged_tree.left = new_node    
    merged_tree.right = sub_tree
    return merged_tree
    

#def simplified_data(num_train, num_dev, num_test):
def simplified_data(num_train, num_dev):
    rndstate = random.getstate()
    random.seed(0)
    #trees = loadTrees('train') + loadTrees('dev') + loadTrees('test')
    pos_trees = loadTrees('training_data.pos.treerevised.txt') 
    neg_trees = loadTrees('training_data.neg.treerevised.txt')
    test_trees = loadTrees('testing_data.txt.treerevised.txt', predicting_test=True )

    #filter extreme trees
    #pos_trees = [t for t in trees if t.root.label==4]
    #neg_trees = [t for t in trees if t.root.label==0]

    #binarize labels
    #binarize_labels(pos_trees)
    #binarize_labels(neg_trees)
    #################
    assign_one_labels(pos_trees)
    assign_zero_labels(neg_trees)
    #################
    
    #split into train, dev, test
    print "len of pos", len(pos_trees), "len of neg", len(neg_trees), "len of test", len(test_trees)
    pos_trees = sorted(pos_trees, key=lambda t: len(t.get_words()))
    neg_trees = sorted(neg_trees, key=lambda t: len(t.get_words()))
    num_train/=2
    num_dev/=2
    #num_test/=2
    train = pos_trees[:num_train] + neg_trees[:num_train]
    dev = pos_trees[num_train : num_train+num_dev] + neg_trees[num_train : num_train+num_dev]
    #test = pos_trees[num_train+num_dev : num_train+num_dev+num_test] + neg_trees[num_train+num_dev : num_train+num_dev+num_test]
    test = test_trees
    random.shuffle(train)
    random.shuffle(dev)
    #random.shuffle(test)
    random.setstate(rndstate)


    return train, dev, test


def binarize_labels(trees):
    def binarize_node(node, _):
        if node.label<2:
            node.label = 0
        elif node.label>2:
            node.label = 1
    for tree in trees:
        leftTraverse(tree.root, binarize_node, None)
        tree.labels = get_labels(tree.root)

def assign_one_labels(trees):
    for tree in trees:
        tree.label = 1

def assign_zero_labels(trees):
    for tree in trees:
        tree.label = 0



if __name__=="__main__":
    #pos_trees = loadTrees('training_data.pos.treerevised.txt') 
    #neg_trees = loadTrees('training_data.neg.treerevised.txt')
    #test_trees = loadTrees('testing_data.txt.treerevised.txt', predicting_test=True )
    #test_trees = loadTrees('testing_data.txt.treerevised.txt')
    #print len(pos_trees)
    #print len(neg_trees)
    #print len(test_trees)
    pass
