from collections import namedtuple
from collections import Counter, OrderedDict, defaultdict
from nltk import Tree
import re
import numpy as np

# this function reads in a textfile and fixes an issue with "\\"
def filereader(path): 
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\","")


def tokens_from_treestring(s):
    """extract the tokens from a sentiment tree"""
    return re.sub(r"\([0-9] |\)", "", s).split()

def transitions_from_treestring(s):
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)
    return list(map(int, s.split()))

def examplereader(path, lower=False):
    """Returns all examples in a file one by one."""
    # A simple way to define a class is using namedtuple.
    Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])
    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = Tree.fromstring(line)  # use NLTK's Tree
        label = int(line[1])
        trans = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)
  

def load_data():
    """Function to load data."""
    LOWER = False  # we will keep the original casing
    print("Loading data into memory")
    train_data = list(examplereader("trees/train.txt", lower=LOWER))
    dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
    test_data = list(examplereader("trees/test.txt", lower=LOWER))

    print("train set:", len(train_data))
    print("dev set:", len(dev_data))
    print("test set:", len(test_data))
    return train_data, dev_data, test_data



class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__,
                        OrderedDict(self))
    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""
    
    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1
        
    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)    
        
    def build(self, min_freq=0):
        '''
        min_freq: minimum number of occurrences for a word to be included  
                in the vocabulary
        '''
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)   
        
        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)

def create_vocabulary(train_data):
    print("Creating vocabulary...")
    v = Vocabulary()
    for data_set in (train_data,):
        for ex in data_set:
            for token in ex.tokens:
                v.count_token(token)
    v.build()
    print("Vocabulary size:", len(v.w2i))
    return v

def build_pt_vocab(embed_f):
    print("Creating pretrained voculabary...")
    v = Vocabulary()
    vectors = []
    vec_length = len(embed_f.readline().split()[1:])
    vectors.append([0]* vec_length)
    vectors.append([0] * vec_length)
    for line in embed_f:
        word = line.split()[0]
        v.count_token(word)

        vector = line.split()[1:]
        vector = [float(weight) for weight in vector]
        vectors.append(vector)

    v.build()
    print("Vocabulary size:", len(v.w2i))
    vectors = np.stack(vectors, axis=0)
    return v, vectors