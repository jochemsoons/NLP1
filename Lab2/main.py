import re
import random
import time
import math
import numpy as np
import nltk
# import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
import torch
from torch import nn
from torch import optim

from data import load_data, create_vocabulary
from models import BOW
from utils import set_seed, print_parameters

def prepare_example(example, vocab):
    """
    Map tokens to their IDs for a single example
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [vocab.w2i.get(t, 0) for t in example.tokens]
    
    x = torch.LongTensor([x])
    x = x.to(device)
    
    y = torch.LongTensor([example.label])
    y = y.to(device)
    
    return x, y

def simple_evaluate(model, data, prep_fn=prepare_example, **kwargs):
    """Accuracy of a model on given data set."""
    correct = 0
    total = 0
    model.eval()  # disable dropout (explained later)

    for example in data:
        
        # convert the example input and label to PyTorch tensors
        x, target = prep_fn(example, model.vocab)

        # forward pass without backpropagation (no_grad)
        # get the output from the neural network for input x
        with torch.no_grad():
            logits = model(x)
        
        # get the prediction
        prediction = logits.argmax(dim=-1)
        
        # add the number of correct predictions to the total correct
        correct += (prediction == target).sum().item()
        total += 1

    return correct, total, correct / float(total)

def get_examples(data, shuffle=True, **kwargs):
    """Shuffle data set and return 1 example at a time (until nothing left)"""
    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch
    for example in data:
        yield example


def train_model(model, optimizer, train_data, dev_data, test_data,
                num_iterations=10000, 
                print_every=1000, eval_every=1000,
                batch_fn=get_examples, 
                prep_fn=prepare_example,
                eval_fn=simple_evaluate,
                batch_size=1, eval_batch_size=None,
                early_stop_crit=3):
    """Train a model."""  
    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss() # loss function
    best_eval = 0.
    best_iter = 0
    
    # store train loss and validation accuracy during training
    # so we can plot them afterwards
    losses = []
    accuracies = []  
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    
    early_stopping_count = 0
    while True:  # when we run out of examples, shuffle and continue
        for batch in batch_fn(train_data, batch_size=batch_size):

            # forward pass
            model.train()
            x, targets = prep_fn(batch, model.vocab)
            logits = model(x)

            B = targets.size(0)  # later we will use B examples per update
            
            # compute cross-entropy loss (our criterion)
            # note that the cross entropy loss function computes the softmax for us
            loss = criterion(logits.view([B, -1]), targets.view(-1))
            train_loss += loss.item()

            # backward pass (tip: check the Introduction to PyTorch notebook)

            # erase previous gradients
            optimizer.zero_grad()
            
            # compute gradients
            loss.backward()

            # update weights - take a small step in the opposite dir of the gradient
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:
                print("Iter %r: loss=%.4f, time=%.2fs" % 
                    (iter_i, train_loss, time.time()-start))
                losses.append(train_loss)
                print_num = 0        
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                _, _, accuracy = eval_fn(model, dev_data, batch_size=eval_batch_size,
                                        batch_fn=batch_fn, prep_fn=prep_fn)
                if len(accuracies) > 0:
                    if accuracies[-1] >= accuracy: 
                        early_stopping_count += 1
                        print("iter %r: dev acc=%.4f. No val improvement: early stopping count=%r" % (iter_i, accuracy, early_stopping_count))  
                else:
                    early_stopping_count = 0
                    print("iter %r: dev acc=%.4f. Val improvement: early stopping count=%r" % (iter_i, accuracy, early_stopping_count))  
                accuracies.append(accuracy)
            
                    
                # save best model parameters
                if accuracy > best_eval:
                    print("new highscore")
                    best_eval = accuracy
                    best_iter = iter_i
                    path = "{}.pt".format(model.__class__.__name__)
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations or early_stopping_count >= early_stop_crit:
                if early_stopping_count >= early_stop_crit:
                    print("Found no improvement for %r validation iterations. Apply early stopping.")
                print("Done training")
                
                # evaluate on train, dev, and test with best model
                print("Loading best model")
                path = "{}.pt".format(model.__class__.__name__)        
                ckpt = torch.load(path)
                model.load_state_dict(ckpt["state_dict"])
                
                _, _, train_acc = eval_fn(
                    model, train_data, batch_size=eval_batch_size, 
                    batch_fn=batch_fn, prep_fn=prep_fn)
                _, _, dev_acc = eval_fn(
                    model, dev_data, batch_size=eval_batch_size,
                    batch_fn=batch_fn, prep_fn=prep_fn)
                _, _, test_acc = eval_fn(
                    model, test_data, batch_size=eval_batch_size, 
                    batch_fn=batch_fn, prep_fn=prep_fn)
                
                print("best model iter {:d}: "
                    "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                        best_iter, train_acc, dev_acc, test_acc))
                
                return losses, accuracies



def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on:", device)

    train_data, dev_data, test_data = load_data()
    v = create_vocabulary(train_data)

    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})

    if args.model == 'BOW':
        # Let's create a model.
        # If everything is in place we can now train our first model!
        bow_model = BOW(len(v.w2i), len(t2i), vocab=v)
        print(bow_model)

        bow_model = bow_model.to(device)

        optimizer = optim.Adam(bow_model.parameters(), lr=0.0005)
        bow_losses, bow_accuracies = train_model(
            bow_model, optimizer, train_data=train_data, dev_data=dev_data, test_data=test_data,
            num_iterations=5000, print_every=1000, eval_every=1000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default='BOW')
    args = parser.parse_args()
    print("#" * 80)
    print("RUNNING ARGUMENTS:")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print("#" * 80)
    train(args)

