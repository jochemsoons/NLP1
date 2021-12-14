from collections import namedtuple
from nltk import Tree
import re


import wget

wget.download('http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip')
# !unzip trainDevTestTrees_PTB.zip

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
  

# Let's load the data into memory.
LOWER = False  # we will keep the original casing
train_data = list(examplereader("trees/train.txt", lower=LOWER))
dev_data = list(examplereader("trees/dev.txt", lower=LOWER))
test_data = list(examplereader("trees/test.txt", lower=LOWER))

print("train", len(train_data))
print("dev", len(dev_data))
print("test", len(test_data))