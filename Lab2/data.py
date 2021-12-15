from collections import namedtuple
from collections import Counter, OrderedDict, defaultdict
from nltk import Tree
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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


def plot_data_statistics():
    print("Printing and plotting dataset statistics...")
    train_set, dev_set, test_set = load_data()
    
    sent_lengths = []
    labels = []
    for data_set in (train_set, dev_set, test_set):
        for ex in data_set:
            sent_lengths.append(len(ex.tokens))
            labels.append(ex.label)

    """NOTE: code for creating bar plots adapted from https://stackoverflow.com/questions/61265580/matplotlib-bar-chart-for-number-of-occurrences"""
    sorted_lenghts = sorted(sent_lengths)
    sorted_counted = Counter(sorted_lenghts)
    
    range_length = list(range(max(sent_lengths))) # Get the largest value to get the range.
    data_series = {}
    for i in range_length:
        data_series[i] = 0 # Initialize series so that we have a template and we just have to fill in the values.

    for key, value in sorted_counted.items():
        data_series[key] = value

    data_series = pd.Series(data_series)
    x_values = data_series.index

    plt.figure()
    plt.bar(x_values, data_series.values, color='blue')
    plt.title("Distribution of sentence lengths in SST dataset")
    plt.xlabel("Sentence length")
    plt.ylabel("Number of occurences in dataset")
    plt.tight_layout()

    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig('./plots/sent_lengths')

    sorted_labels = sorted(labels)
    sorted_counted = Counter(sorted_labels)
    
    range_length = list(range(max(labels))) # Get the largest value to get the range.
    data_series = {}
    for i in range_length:
        data_series[i] = 0 # Initialize series so that we have a template and we just have to fill in the values.

    for key, value in sorted_counted.items():
        data_series[key] = value

    data_series = pd.Series(data_series)
    x_values = data_series.index
    y_values = [(value / sum(data_series.values)) for value in data_series.values]
    plt.figure()
    plt.bar(x_values, y_values, color='blue')
    plt.title("Distribution of sentiment labels in SST dataset")
    plt.xlabel("Sentiment label")
    x_labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    plt.xticks(x_values, x_labels, rotation=45)
    plt.ylabel("Percentage in dataset")
    plt.tight_layout()

    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig('./plots/sentiments_distr')
    
def split_sentence_lengths(data_set):
    sents_0_10 = []
    sents_10_20 = []
    sents_20_30 = []
    sents_30_up = []
    for ex in data_set:
        if len(ex.tokens) <= 10:
            print("0-10", len(ex.tokens))
            sents_0_10.append(ex)
        elif 10 < len(ex.tokens) <= 20:
            print("10-20", len(ex.tokens))
            sents_10_20.append(ex)
        elif 20 < len(ex.tokens) <= 30:
            print("20-30", len(ex.tokens))
            sents_20_30.append(ex)
        elif len(ex.tokens) > 30:
            print("30-up", len(ex.tokens))
            sents_30_up.append(ex)
    return sents_0_10, sents_10_20, sents_20_30, sents_30_up
    