import torch
import numpy as np
import pickle 
import os

def set_seed(seed):
    print("Setting random seed:", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False


# Here we print each parameter name, shape, and if it is trainable.
def print_parameters(model):
  total = 0
  for name, p in model.named_parameters():
    total += np.prod(p.shape)
    print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)), p.requires_grad))
  print("\nTotal number of parameters: {}\n".format(total))


# Function to dump a variable as a pickle file.
def pdump(values, filename, dirname='./pickle_dumps/') :
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print("Dumping " + filename)
    pickle.dump(values, open(os.path.join(dirname + filename + '_pdump.pkl'), 'wb'))

# Function to load a variable from a pickle file.
def pload(filename, dirname='./pickle_dumps/') :
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = os.path.join(dirname + filename + '_pdump.pkl')
    if not os.path.isfile(file) :
        raise FileNotFoundError(file + " doesn't exist")
    else:
        print("Loading " + filename)
        return pickle.load(open(file, 'rb'))
