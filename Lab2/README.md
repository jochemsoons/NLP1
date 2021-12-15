# NLP1 Practical II

## Sentence Sentiment Classification with Neural Models

This repository contains the code for the second Lab assignment of the NLP1 course at the university of Amsterdam, 2021.

**Students:**

* Jochem Soons:  #11327030
* Darius Barsony: #11234342

**TA**:

* Floor van Lieshout

#### Note:

We transformed the code written in the colab notebook to a repository containing multiple files, as we found this a more practical way of working on this project and running the code. We have also included the .ipynb notebook in this folder in case this might help with grading our work.

## Files included

The files where you can find our code:
- main.py: this file contains the code pipeline for running all experiments, including the train_model() function.
- data.py: this file contains all code for loading in the data, creating vocabularies, creating the shuffled input, splitting a dataset in bins of different lengths and extracting subtrees from sentences.
- models.py: this file contains the code for all our model implementations, including: BOW, CBOW, DeepCBOW, PT_DeepCBOW, LSTM, TreeLSTM and the Child-Sum TreeLSTM.
- utils.py: this file contains some small helper functions such as setting the seed, printing model parameters and dumping/loading pickle files.
- plot_sentence_exp.py: this file contains the code to create the plot belonging to the sentence length experiment (more about this below when we show some examples to run our code).

## Instructions of usage

We provide an environment file in environment.yml which you might use to run our code:

    conda env create -f environment.yml
    conda activate nlp1
    
Subsequently, all our experiments can be reproduced by running main.py and specifying command line arguments

main.py has the following parsing arguments:

- `--model` defines the model being deployed. Can be 'BOW', 'CBOW', 'DeepCBOW', 'pt_DeepCBOW', 'LSTM', 'TreeLSTM', or 'all'.
- `--pt_embed` defines the type of pretrained word embedding used. Can be either 'glove' or 'w2v'. Default value is 'w2v'.
- `--num_iterations` defines the number of training iterations. Default value is 30.000.
- `--print_every` defines at what iteration interval scores are printed. Default value is 500.
- `--eval_every` defines at what iteration interval scores evaluation is performed on the validation set. Default value is 500.
- `--batch_size` defines the batch size used for the LSTM models. Default value is 25. Note: only applies to the LSTM models.
- `--early_stopping` if this argument is added early stopping is applied to training. Training is stopped when evaluation performance does not improve for 3 intervals. Note: we did not use this in obtaining our results.
- `--childsum` if this argument is added the Child-Sum Tree-LSTM will be implemented. Note: model needs to be 'TreeLSTM' for this argument to work.
- `--supervision` if this argument is added the Tree-LSTM will be implemented with node supervision. Note: model needs to be 'TreeLSTM' for this argument to work.
- `--keep_ckpts` if this argument is added the torch checkpoints of models that are used to extract the best performing model are saved in a directory. Otherwise, they are automatically deleted to limit disk storage.
- `--random_permute` if this argument is added, then all sentences in the training and validation dataset are randomly permuted. Note: should only be used with LSTM as model.
- `--plot_data_statistics` if this argument is added, additional plots are created that highlight some statistics of our dataset.
- `--split_sentence_length` if this argument is added, then the test set is split into four seperate sets of different sentence lengths.

## Some examples

In all examples below results are averaged over three separate runs using different random seeds.

To reproduce results for all models using word2vec embeddings, run:
```
python3 main.py --model all --pt_embed w2v
```

To reproduce results for the experiment of **randomly shuffling input** to the regular LSTM model using GloVe embeddings, run:
```
python3 main.py --model LSTM --pt_embed glove --random_permute
```

To reproduce results for all models when evaluated on test sets of **different sentence lengths** (using word2vec embeddings), run:
```
python3 main.py --model all --pt_embed w2v --split_sentence_lengths
```

Subsquently, to obtain the grouped barplot we included in our report, run:
```
python3 plot_sentence_exp.py
```


To reproduce results of the **Tree-LSTM using node supervision**, run:
```
python3 main.py --model TreeLSTM --supervision
```

To reproduce results of the **Child-Sum Tree-LSTM**, run:
```
python3 main.py --model TreeLSTM --childsum
```



