import random
import time
import os
import numpy as np
from collections import OrderedDict
import argparse
import torch
from torch import nn
from torch import optim

from data import load_data, create_vocabulary, build_pt_vocab, plot_data_statistics, split_sentence_lengths
from models import BOW, CBOW, DeepCBOW, PTDeepCBOW, LSTMClassifier, TreeLSTMClassifier
from utils import set_seed, print_parameters, pdump, pload

def prepare_example(example, vocab, random_permute=False):
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

def get_minibatch(data, batch_size=25, shuffle=True):
    """Return minibatches, optional shuffling"""
    
    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch
    
    batch = []
    
    # yield minibatches
    for example in data:
        batch.append(example)
        
        if len(batch) == batch_size:
            yield batch
            batch = []
        
    # in case there is something left
    if len(batch) > 0:
        yield batch

def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))

def prepare_minibatch(mb, vocab, random_permute=False):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])
    
    # Random permute sentence before padding.
    if random_permute:
        x = [pad(list(np.random.permutation([vocab.w2i.get(t, 0) for t in ex.tokens])), maxlen) for ex in mb]
    else:
        x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    x = torch.LongTensor(x)
    x = x.to(device)
    
    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)
    
    return x, y

def evaluate(model, data, 
             batch_fn=get_minibatch, prep_fn=prepare_minibatch,
             batch_size=16, random_permute=False):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval()  # disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab, random_permute=random_permute)
        with torch.no_grad():
            logits = model(x)
        
        predictions = logits.argmax(dim=-1).view(-1)
        
        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total)

def prepare_treelstm_minibatch(mb, vocab, random_permute=False):
    """
    Returns sentences reversed (last word first)
    Returns transitions together with the sentences.  
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = len(mb)
    maxlen = max([len(ex.tokens) for ex in mb])
        
    # vocab returns 0 if the word is not there
    # NOTE: reversed sequence!
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]
    
    x = torch.LongTensor(x)
    x = x.to(device)
    
    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(device)
    
    maxlen_t = max([len(ex.transitions) for ex in mb])
    transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
    transitions = np.array(transitions)
    transitions = transitions.T  # time-major
    
    return (x, transitions), y

def train_model(model, optimizer, train_data, dev_data, test_data,
                num_iterations=10000, 
                print_every=1000, eval_every=1000,
                batch_fn=get_examples, 
                prep_fn=prepare_example,
                eval_fn=simple_evaluate,
                batch_size=1, eval_batch_size=None,
                early_stopping=False,
                random_permute=False):
    """Train a model."""  
    iter_i = 0
    train_loss = 0.
    print_num = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss() # loss function
    best_eval = 0.
    best_iter = 0
    if early_stopping:
        early_stop_crit = 3
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
            x, targets = prep_fn(batch, model.vocab, random_permute=random_permute)
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
                                        batch_fn=batch_fn, prep_fn=prep_fn, random_permute=random_permute)
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
                    if not os.path.exists('./model_ckpts'):
                        os.makedirs('./model_ckpts')
                    path = "./model_ckpts/{}_{}.pt".format(model.__class__.__name__, start)
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations or (early_stopping and early_stopping_count >= early_stop_crit):
                if early_stopping and early_stopping_count >= early_stop_crit:
                    print("Found no improvement for %r validation iterations. Apply early stopping.")
                print("Done training")
                
                # evaluate on train, dev, and test with best model
                print("Loading best model")
                path = "./model_ckpts/{}_{}.pt".format(model.__class__.__name__, start)        
                ckpt = torch.load(path)
                model.load_state_dict(ckpt["state_dict"])
                
                # Remove checkpoint file.
                if os.path.exists(path) and not args.keep_ckpts:
                    os.remove(path)
        
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
                
                return losses, accuracies, best_iter, train_acc, dev_acc, test_acc



def train(args, seed, device, train_data, dev_data, test_data):
    # Set random seed for reproducibility.
    if seed:
        set_seed(seed)

    v = create_vocabulary(train_data)
    i2t = ["very negative", "negative", "neutral", "positive", "very positive"]
    t2i = OrderedDict({p : i for p, i in zip(i2t, range(len(i2t)))})

    if args.model == 'BOW':
        bow_model = BOW(len(v.w2i), len(t2i), vocab=v)
        print(bow_model)
        bow_model = bow_model.to(device)
        optimizer = optim.Adam(bow_model.parameters(), lr=0.0005)
        return train_model(bow_model, optimizer, 
            train_data=train_data, dev_data=dev_data, test_data=test_data,
            num_iterations=args.num_iterations, print_every=args.print_every, eval_every=args.eval_every, early_stopping=args.early_stopping)

    elif args.model == 'CBOW':
        cbow_model = CBOW(vocab_size=len(v.w2i),
                  embedding_dim=300, 
                  num_classes=5,
                  vocab=v)
        print(cbow_model)
        cbow_model = cbow_model.to(device)
        optimizer = optim.Adam(cbow_model.parameters(), lr=0.0005)
        return train_model(cbow_model, optimizer, 
            train_data=train_data, dev_data=dev_data, test_data=test_data,
            num_iterations=args.num_iterations, print_every=args.print_every, eval_every=args.eval_every, early_stopping=args.early_stopping)

    elif args.model == 'DeepCBOW':
        dcbow_model = DeepCBOW(vocab_size=len(v.w2i),
                  embedding_dim=300, 
                  layer_dim=100,
                  num_classes=5,
                  vocab=v)
        print(dcbow_model)
        dcbow_model = dcbow_model.to(device)
        optimizer = optim.Adam(dcbow_model.parameters(), lr=0.0005)
        return train_model(dcbow_model, optimizer, 
            train_data=train_data, dev_data=dev_data, test_data=test_data,
            num_iterations=args.num_iterations, print_every=args.print_every, eval_every=args.eval_every, early_stopping=args.early_stopping)

    else:
        glove_file = open("glove.840B.300d.sst.txt")
        word2vec_file = open("googlenews.word2vec.300d.txt")
        if args.pt_embed == 'w2v':
            pretrained_v, vectors = build_pt_vocab(word2vec_file)
        elif args.pt_embed == 'glove':
            pretrained_v, vectors = build_pt_vocab(glove_file)
        
        if args.model == 'pt_DeepCBOW':
            pt_deep_cbow_model = PTDeepCBOW(vocab_size=len(pretrained_v.w2i),
                                embedding_dim=300,
                                hidden_dim=100,
                                output_dim=5,
                                vocab=pretrained_v)
            print(pt_deep_cbow_model)

            # copy pre-trained word vectors into embeddings table
            pt_deep_cbow_model.embed.weight.data.copy_(torch.from_numpy(vectors))
            # disable training the pre-trained embeddings
            pt_deep_cbow_model.embed.weight.requires_grad = False
            # move model to specified device
            pt_deep_cbow_model = pt_deep_cbow_model.to(device)

            # train the model
            optimizer = optim.Adam(pt_deep_cbow_model.parameters(), lr=0.0005)
            return train_model(pt_deep_cbow_model, optimizer,
                train_data=train_data, dev_data=dev_data, test_data=test_data, 
                num_iterations=args.num_iterations, print_every=args.print_every, eval_every=args.eval_every, early_stopping=args.early_stopping)

        elif args.model == 'LSTM':
            lstm_model = LSTMClassifier(len(pretrained_v.w2i), 300, 168, len(t2i), pretrained_v)

            # copy pre-trained vectors into embeddings table
            with torch.no_grad():
                lstm_model.embed.weight.data.copy_(torch.from_numpy(vectors))
                lstm_model.embed.weight.requires_grad = False

            print(lstm_model)
            print_parameters(lstm_model)  
            lstm_model = lstm_model.to(device)
            optimizer = optim.Adam(lstm_model.parameters(), lr=2e-4)

            return train_model(lstm_model, optimizer, 
                train_data=train_data, dev_data=dev_data, test_data=test_data, 
                num_iterations=args.num_iterations, print_every=args.print_every, eval_every=args.eval_every,
                batch_size=args.batch_size,
                batch_fn=get_minibatch, 
                prep_fn=prepare_minibatch,
                eval_fn=evaluate,
                early_stopping=args.early_stopping,
                random_permute=args.random_permute)

        elif args.model == 'TreeLSTM':
            tree_model = TreeLSTMClassifier(
                len(pretrained_v.w2i), 300, 150, len(t2i), pretrained_v)

            with torch.no_grad():
                tree_model.embed.weight.data.copy_(torch.from_numpy(vectors))
                tree_model.embed.weight.requires_grad = False

            print(tree_model)
            print_parameters(tree_model)
            tree_model = tree_model.to(device)
            optimizer = optim.Adam(tree_model.parameters(), lr=2e-4)

            return train_model(tree_model, optimizer, 
                train_data=train_data, dev_data=dev_data, test_data=test_data, 
                num_iterations=args.num_iterations, print_every=args.print_every, eval_every=args.eval_every,
                prep_fn=prepare_treelstm_minibatch,
                eval_fn=evaluate,
                batch_fn=get_minibatch,
                batch_size=args.batch_size, eval_batch_size=args.batch_size,
                early_stopping=args.early_stopping)
    
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default='BOW', choices=['BOW', 'CBOW', 'DeepCBOW', 'pt_DeepCBOW', 'LSTM', 'TreeLSTM', 'all'])
    parser.add_argument('--pt_embed', type = str, default='w2v', choices=['w2v', 'glove'])
    parser.add_argument('--num_iterations', type=int, default=30000)
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=25)
    parser.add_argument('--early_stopping', default=False, action='store_true')
    parser.add_argument('--keep_ckpts', default=False, action='store_true')
    parser.add_argument('--random_permute', default=False, action='store_true')
    parser.add_argument('--plot_data_statistics', default=False, action='store_true')
    parser.add_argument('--split_sentence_lengths', default=False, action='store_true')
    args = parser.parse_args()

    # Print parsing arguments.
    print("#" * 80)
    print("RUNNING ARGUMENTS:")
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print("#" * 80)

    # Random permute of input can only be performed for LSTM model:
    if args.random_permute and (args.model != 'LSTM' or args.split_sentence_lengths):
        raise AssertionError("Random permute experiment can only be performed for LSTM model (single run)")

    if args.plot_data_statistics:
        plot_data_statistics()

    # Setup device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on:", device)
    
    # Load data.
    train_data, dev_data, test_data = load_data()

    # Single model, run 4 times with different seeds.
    if args.model != 'all' and not args.split_sentence_lengths:
        scores = []
        best_iters = []
        for i, seed in enumerate([42, 43, 44]):
            loss_list, acc_list, best_iter, train_acc, dev_acc, test_acc = train(args, seed, device, train_data, dev_data, test_data)
            print("\nModel:", args.model)
            print("Run {}/{}, seed: {}".format(i+1, 3, seed))
            print("Train acc: {:.4f}, Val acc: {:.4f}, test acc: {:.4f}".format(train_acc, dev_acc, test_acc))
            print("Best iteration:{}\n".format(best_iter))
            scores.append(test_acc*100)
            best_iters.append(best_iter)
        print("MODEL RESULTS:", args.model)
        print("ACC: {:.2f}, std: {:.2f}".format(np.mean(scores), np.std(scores)))
        print("BEST ITER: {:.0f}".format(np.mean(best_iters)))
    
    # Run all models 3 times with different seeds.
    elif args.model == 'all' and not args.split_sentence_lengths:
        for model in ['BOW', 'CBOW', 'DeepCBOW', 'pt_DeepCBOW', 'LSTM', 'TreeLSTM']:
            scores = []
            best_iters = []
            for i, seed in enumerate([42, 43, 44]):
                args.model = model
                if model in ['BOW', 'CBOW', 'DeepCBOW', 'pt_DeepCBOW']:
                    args.num_iterations = 30000
                else:
                    args.num_iterations = 25000
                loss_list, acc_list, best_iter, train_acc, dev_acc, test_acc = train(args, seed, device, train_data, dev_data, test_data)
                scores.append(test_acc*100)
                best_iters.append(best_iter)
                print("\nModel:", model)
                print("Run {}/{}, seed: {}".format(i+1, 3, seed))
                print("Train acc: {:.4f}, Val acc: {:.4f}, test acc: {:.4f}".format(train_acc, dev_acc, test_acc))
                print("Best iteration:{}\n".format(best_iter))
            print("MODEL RESULTS:", model)
            print("ACC: {:.2f}, std: {:.2f}".format(np.mean(scores), np.std(scores)))
            print("BEST ITER: {:.0f}".format(np.mean(best_iters)))

    # Run all models 3 * 4 times (3 seeds, 4 datasets of different sentence lengths) 
    elif args.model == 'all' and args.split_sentence_lengths:
        train_sets = split_sentence_lengths(train_data)
        dev_sets  = split_sentence_lengths(dev_data)
        test_sets = split_sentence_lengths(test_data)
        
        for i in range(4):
            train_data = train_sets[i]
            dev_data = train_sets[i]
            test_data = train_sets[i]
            
            for model in ['BOW', 'CBOW', 'DeepCBOW', 'pt_DeepCBOW', 'LSTM', 'TreeLSTM']:
                scores = []
                best_iters = []
                for seed in [42, 43, 44]:
                    args.model = model
                    if model in ['BOW', 'CBOW', 'DeepCBOW', 'pt_DeepCBOW']:
                        args.num_iterations = 30000
                    else:
                        args.num_iterations = 25000
                    loss_list, acc_list, best_iter, train_acc, dev_acc, test_acc = train(args, seed, device, train_data, dev_data, test_data)
                    scores.append(test_acc*100)
                    best_iters.append(best_iter)

                print("MODEL RESULTS:", model)
                print("SENTENCE SPLIT:", i)
                print("ACC: {:.2f}, std: {:.2f}".format(np.mean(scores), np.std(scores)))
                print("BEST ITER: {:.0f}".format(np.mean(best_iters)))
                pdump(scores, "scores_{}_{}".format(model, i))

            
