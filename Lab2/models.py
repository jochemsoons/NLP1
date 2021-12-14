import torch
from torch import nn
import math

class BOW(nn.Module):
  """A simple bag-of-words model"""

  def __init__(self, vocab_size, embedding_dim, vocab):
    super(BOW, self).__init__()
    self.vocab = vocab
    
    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)
    
    # this is a trainable bias term
    self.bias = nn.Parameter(torch.zeros(embedding_dim), requires_grad=True)        

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)
    
    # the output is the sum across the time dimension (1)
    # with the bias term added
    logits = embeds.sum(1) + self.bias
    return logits


class CBOW(nn.Module):
  """The continuous bag-of-words model"""

  def __init__(self, vocab_size, embedding_dim, num_classes, vocab):
    super(CBOW, self).__init__()
    self.vocab = vocab
    
    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)
    self.W = nn.Linear(embedding_dim, num_classes)
            

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)
    # Take the sum across the time dimension (1)
    logits_sum = embeds.sum(1)

    # Output our logits using W.
    logits = self.W(logits_sum)

    return logits

class DeepCBOW(nn.Module):
  """The Deep continuous bag-of-words model"""

  def __init__(self, vocab_size, embedding_dim, layer_dim, num_classes, vocab):
    super(DeepCBOW, self).__init__()
    self.vocab = vocab
    
    # this is a trainable look-up table with word embeddings
    self.embed = nn.Embedding(vocab_size, embedding_dim)
    self.output_layer = nn.Sequential(
                            nn.Linear(embedding_dim, layer_dim),
                            nn.Tanh(),
                            nn.Linear(layer_dim, layer_dim),
                            nn.Tanh(),
                            nn.Linear(layer_dim, num_classes)
    )
                

  def forward(self, inputs):
    # this is the forward pass of the neural network
    # it applies a function to the input and returns the output

    # this looks up the embeddings for each word ID in inputs
    # the result is a sequence of word embeddings
    embeds = self.embed(inputs)
    # Take the sum across the time dimension (1)
    logits_sum = embeds.sum(1)

    # Output our logits using the output layers.
    logits = self.output_layer(logits_sum)

    return logits

class PTDeepCBOW(DeepCBOW):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(PTDeepCBOW, self).__init__(
            vocab_size, embedding_dim, hidden_dim, output_dim, vocab)


class MyLSTMCell(nn.Module):
    """Our own LSTM cell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(MyLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        self.L_ii = nn.Linear(input_size, hidden_size, bias=bias)
        self.L_hi = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        self.L_if = nn.Linear(input_size, hidden_size, bias=bias)
        self.L_hf = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.L_ig = nn.Linear(input_size, hidden_size, bias=bias)
        self.L_hg = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.L_io = nn.Linear(input_size, hidden_size, bias=bias)
        self.L_ho = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)  

    def forward(self, input_, hx, mask=None):
        """
        input is (batch, input_size)
        hx is ((batch, hidden_size), (batch, hidden_size))
        """
        prev_h, prev_c = hx

        # project input and prev state

        # main LSTM computation    
        i = self.sigmoid(self.L_ii(input_) + self.L_hi(prev_h))
        f = self.sigmoid(self.L_if(input_) + self.L_hf(prev_h))
        g = self.tanh(self.L_ig(input_) + self.L_hg(prev_h)) 
        o = self.sigmoid(self.L_io(input_) + self.L_ho(prev_h))

        c = f * prev_c + i * g
        h = o * self.tanh(c)
        return h, c
    
    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size)


class LSTMClassifier(nn.Module):
    """Encodes sentence with an LSTM and projects final hidden state"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(LSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.rnn = MyLSTMCell(embedding_dim, hidden_dim)
        
        self.output_layer = nn.Sequential(     
            nn.Dropout(p=0.5),  # explained later
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        
        B = x.size(0)  # batch size (this is 1 for now, i.e. 1 single example)
        T = x.size(1)  # timesteps (the number of words in the sentence)
        
        input_ = self.embed(x)

        # here we create initial hidden states containing zeros
        # we use a trick here so that, if input is on the GPU, then so are hx and cx
        hx = input_.new_zeros(B, self.rnn.hidden_size)
        cx = input_.new_zeros(B, self.rnn.hidden_size)
        
        # process input sentences one word/timestep at a time
        # input is batch-major (i.e., batch size is the first dimension)
        # so the first word(s) is (are) input_[:, 0]
        outputs = []   
        for i in range(T):
            hx, cx = self.rnn(input_[:, i], (hx, cx))
            outputs.append(hx)
        
        # if we have a single example, our final LSTM state is the last hx
        if B == 1:
            final = hx
        else:
            #
            # This part is explained in next section, ignore this else-block for now.
            #
            # We processed sentences with different lengths, so some of the sentences
            # had already finished and we have been adding padding inputs to hx.
            # We select the final state based on the length of each sentence.
            
            # two lines below not needed if using LSTM from pytorch
            outputs = torch.stack(outputs, dim=0)           # [T, B, D]
            outputs = outputs.transpose(0, 1).contiguous()  # [B, T, D]
            
            # to be super-sure we're not accidentally indexing the wrong state
            # we zero out positions that are invalid
            pad_positions = (x == 1).unsqueeze(-1)
            
            outputs = outputs.contiguous()      
            outputs = outputs.masked_fill_(pad_positions, 0.)
                
            mask = (x != 1)  # true for valid positions [B, T]
            lengths = mask.sum(dim=1)                 # [B, 1]

            indexes = (lengths - 1) + torch.arange(B, device=x.device, dtype=x.dtype) * T
            final = outputs.view(-1, self.hidden_dim)[indexes]  # [B, D]
            
        # we use the last hidden state to classify the sentence
        logits = self.output_layer(final)
        return logits


class TreeLSTMCell(nn.Module):
    """A Binary Tree LSTM cell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(TreeLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.reduce_layer = nn.Linear(2 * hidden_size, 5 * hidden_size)
        self.dropout_layer = nn.Dropout(p=0.25)

        self.reset_parameters()

    def reset_parameters(self):
        """This is PyTorch's default initialization method"""
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)  

    def forward(self, hx_l, hx_r, mask=None):
        """
        hx_l is ((batch, hidden_size), (batch, hidden_size))
        hx_r is ((batch, hidden_size), (batch, hidden_size))    
        """
        prev_h_l, prev_c_l = hx_l  # left child
        prev_h_r, prev_c_r = hx_r  # right child

        B = prev_h_l.size(0)

        # we concatenate the left and right children
        # you can also project from them separately and then sum
        children = torch.cat([prev_h_l, prev_h_r], dim=1)
        
        # project the combined children into a 5D tensor for i,fl,fr,g,o
        # this is done for speed, and you could also do it separately
        proj = self.reduce_layer(children)  # shape: B x 5D

        # each shape: B x D
        i, f_l, f_r, g, o = torch.chunk(proj, 5, dim=-1)

        # main Tree LSTM computation
        
        # YOUR CODE HERE
        # You only need to complete the commented lines below.

        # The shape of each of these is [batch_size, hidden_size]

        i = torch.sigmoid(i)
        f_l = torch.sigmoid(f_l)    
        f_r = torch.sigmoid(f_r)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f_l * prev_c_l + f_r * prev_c_r + i * g
        h = o * torch.tanh(c)
        
        return h, c
    
    def __repr__(self):
        return "{}({:d}, {:d})".format(
            self.__class__.__name__, self.input_size, self.hidden_size)

# Helper functions for batching and unbatching states
# For speed we want to combine computations by batching, but 
# for processing logic we want to turn the output into lists again
# to easily manipulate.

def batch(states):
    """
    Turns a list of states into a single tensor for fast processing. 
    This function also chunks (splits) each state into a (h, c) pair"""
    return torch.cat(states, 0).chunk(2, 1)

def unbatch(state):
    """
    Turns a tensor back into a list of states.
    First, (h, c) are merged into a single state.
    Then the result is split into a list of sentences.
    """
    return torch.split(torch.cat(state, 1), 1, 0)


class TreeLSTM(nn.Module):
    """Encodes a sentence using a TreeLSTMCell"""

    def __init__(self, input_size, hidden_size, bias=True):
        """Creates the weights for this LSTM"""
        super(TreeLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.reduce = TreeLSTMCell(input_size, hidden_size)

        # project word to initial c
        self.proj_x = nn.Linear(input_size, hidden_size)
        self.proj_x_gate = nn.Linear(input_size, hidden_size)
        
        self.buffers_dropout = nn.Dropout(p=0.5)

    def forward(self, x, transitions):
        """
        WARNING: assuming x is reversed!
        :param x: word embeddings [B, T, E]
        :param transitions: [2T-1, B]
        :return: root states
        """
        SHIFT = 0
        REDUCE = 1

        B = x.size(0)  # batch size
        T = x.size(1)  # time

        # compute an initial c and h for each word
        # Note: this corresponds to input x in the Tai et al. Tree LSTM paper.
        # We do not handle input x in the TreeLSTMCell itself.
        buffers_c = self.proj_x(x)
        buffers_h = buffers_c.tanh()
        buffers_h_gate = self.proj_x_gate(x).sigmoid()
        buffers_h = buffers_h_gate * buffers_h
        
        # concatenate h and c for each word
        buffers = torch.cat([buffers_h, buffers_c], dim=-1)

        D = buffers.size(-1) // 2

        # we turn buffers into a list of stacks (1 stack for each sentence)
        # first we split buffers so that it is a list of sentences (length B)
        # then we split each sentence to be a list of word vectors
        buffers = buffers.split(1, dim=0)  # Bx[T, 2D]
        buffers = [list(b.squeeze(0).split(1, dim=0)) for b in buffers]  # BxTx[2D]

        # create B empty stacks
        stacks = [[] for _ in buffers]

        # t_batch holds 1 transition for each sentence
        for t_batch in transitions:

            child_l = []  # contains the left child for each sentence with reduce action
            child_r = []  # contains the corresponding right child

            # iterate over sentences in the batch
            # each has a transition t, a buffer and a stack
            for transition, buffer, stack in zip(t_batch, buffers, stacks):
                if transition == SHIFT:
                    stack.append(buffer.pop())
                elif transition == REDUCE:
                    assert len(stack) >= 2, \
                        "Stack too small! Should not happen with valid transition sequences"
                    child_r.append(stack.pop())  # right child is on top
                    child_l.append(stack.pop())

            # if there are sentences with reduce transition, perform them batched
            if child_l:
                reduced = iter(unbatch(self.reduce(batch(child_l), batch(child_r))))
                for transition, stack in zip(t_batch, stacks):
                    if transition == REDUCE:
                        stack.append(next(reduced))

        final = [stack.pop().chunk(2, -1)[0] for stack in stacks]
        final = torch.cat(final, dim=0)  # tensor [B, D]

        return final


class TreeLSTMClassifier(nn.Module):
    """Encodes sentence with a TreeLSTM and projects final hidden state"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, vocab):
        super(TreeLSTMClassifier, self).__init__()
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)
        self.treelstm = TreeLSTM(embedding_dim, hidden_dim)
        self.output_layer = nn.Sequential(     
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )

    def forward(self, x):
        
        # x is a pair here of words and transitions; we unpack it here.
        # x is batch-major: [B, T], transitions is time major [2T-1, B]
        x, transitions = x
        emb = self.embed(x)
        
        # we use the root/top state of the Tree LSTM to classify the sentence
        root_states = self.treelstm(emb, transitions)

        # we use the last hidden state to classify the sentence
        logits = self.output_layer(root_states)
        return logits