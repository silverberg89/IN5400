from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


######################################################################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super(imageCaptionModel, self).__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network
        
        Args:
            config: Dictionary holding neural network configuration

        Returns:
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputLayer : An instance of nn.Linear, shape[VggFc7Size, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputLayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config = config
        self.vocabulary_size    = config['vocabulary_size']
        self.embedding_size     = config['embedding_size']
        self.VggFc7Size         = config['VggFc7Size']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers     = config['num_rnn_layers']
        self.cell_type          = config['cellType']

        # ToDo
        self.Embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

        self.inputLayer = nn.Linear(self.VggFc7Size,self.hidden_state_sizes)

        self.rnn = RNN(self.embedding_size,self.hidden_state_sizes,self.num_rnn_layers,self.cell_type) 

        self.outputLayer = nn.Linear(self.hidden_state_sizes,self.vocabulary_size)
        return

    def forward(self, vgg_fc7_features, xTokens, is_train, current_hidden_state=None):
        """
        Args:
            vgg_fc7_features    : Features from the VGG16 network, shape[batch_size, VggFc7Size]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # ToDO
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.
        # use self.rnn to calculate "logits" and "current_hidden_state"

        if current_hidden_state is None:
            state = self.inputLayer(vgg_fc7_features)                                    # 4,64
            state = torch.tanh(state)                                                    # Non-Linearity
            state = torch.unsqueeze(state, 0)                                            # 1,4,64
            initial_hidden_state = torch.zeros(self.num_rnn_layers,xTokens.shape[0],self.hidden_state_sizes)
            # print("initial_hidden_state", np.shape(initial_hidden_state))               # Should be 3,4,64
            for i in range(self.num_rnn_layers):
                initial_hidden_state[i,:,:] = state
            logits, current_hidden_state_out = self.rnn(xTokens, initial_hidden_state, self.outputLayer, self.Embedding, is_train)
        else:
            logits, current_hidden_state_out = self.rnn(xTokens, current_hidden_state, self.outputLayer, self.Embedding, is_train)

        return logits, current_hidden_state_out

######################################################################################################################
class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, cell_type='RNN'):
        super(RNN, self).__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers) 
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells
            
        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size        = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers    = num_rnn_layers
        self.cell_type         = cell_type  
        
        # ToDo
        # Your task is to create a list (self.cells) of type "nn.ModuleList" and populated it with cells of type "self.cell_type".
        
        itt = 0
        self.cells = nn.ModuleList([])
        for cells in range(self.num_rnn_layers):
            if itt == 0:
                if (self.cell_type == str('RNN')):
                    self.cells.append(RNNCell(self.hidden_state_size, self.input_size))
                if (self.cell_type == str('GRU')):
                    self.cells.append(GRUCell(self.hidden_state_size, self.input_size))
            else:
                if (self.cell_type == str('RNN')):
                    self.cells.append(RNNCell(self.hidden_state_size, self.hidden_state_size))
                if (self.cell_type == str('GRU')):
                    self.cells.append(GRUCell(self.hidden_state_size, self.hidden_state_size))
            itt+=1

        return
    
    def iteration(self,initial_hidden_state,x,i):
        """
        Args:
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            x:                     Masked embedding matrix
            i:                     Actual sequence element

        Returns:
            prev:                  Current_hidden_state
        """
        for j in range (self.num_rnn_layers):                                   # Calculate current state by iterating through each cell
            if j == 0:
                curr = initial_hidden_state[j].clone()                          # Nessecary to clone for backward calculations
                initial_hidden_state[j] = self.cells[j].forward(x[:,i,:], curr)
                prev = initial_hidden_state[j].clone()
                
            else:
                curr = initial_hidden_state[j].clone()
                initial_hidden_state[j] = self.cells[j].forward(prev, curr)
                prev = initial_hidden_state[j].clone()
        return prev

    def forward(self, xTokens, initial_hidden_state, outputLayer, Embedding, is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train==True:
            seqLen = xTokens.shape[1] #truncated_backprop_length
        else:
            seqLen = 40 #Max sequence length to be generated
            
        # ToDo
        # While iterate through the (stacked) rnn, it may be easier to use lists instead of indexing the tensors.
        # You can use "list(torch.unbind())" and "torch.stack()" to convert from pytorch tensor to lists and back again.
        # get input embedding vectors
        # Use for loops to run over "seqLen" and "self.num_rnn_layers" to calculate logits
        # Produce outputs
        py_list = []
        device = xTokens.device                                         # Assure correct CUDA / CPU tensor
        initial_hidden_state = initial_hidden_state.to(device)
        if is_train == True:
            x = Embedding(xTokens)
            for i in range (seqLen):                                    # Iterates through all 30 potential words
                prev = self.iteration(initial_hidden_state,x,i)
                logit = outputLayer(prev)                               # Calculate logit for each word
                py_list.append(logit)                                   # Saves the currect logits [batch_size, vocabulary_size]
            logits = torch.stack(py_list, dim=1)                        # Stack all logits along dim 1 [batch_size, seqLen, vocabulary_size]
            
        else:
            
            batch_size = np.shape(initial_hidden_state[0])[0]
            tokens = torch.ones([batch_size,seqLen+1], dtype=torch.long).to(device)
            x = Embedding(tokens)                                       # [batch_size, seqLen, embedding_size]
            for i in range (seqLen):
                prev = self.iteration(initial_hidden_state,x,i)
                logit = outputLayer(prev)
                values, indices = torch.max(logit, 1)                   # Gives max values and thier indices
                x[:,i+1,:] = Embedding(indices)                         # Insert masked values
                py_list.append(logit)
            logits = torch.stack(py_list, dim=1)
        
        return logits, initial_hidden_state
        
########################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(GRUCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight_u: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight_r: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean. 

            self.weight: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean. 

            self.bias_u: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias_r: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero. 

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        N                       = hidden_state_size+input_size
        M                       = hidden_state_size

        # TODO:
        self.weight_u = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, np.sqrt(1/N),size=(N,M))))
        self.bias_u   = torch.nn.Parameter(torch.zeros((1,M)))

        self.weight_r = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, np.sqrt(1/N),size=(N,M))))
        self.bias_r   = torch.nn.Parameter(torch.zeros((1,M)))

        self.weight = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, np.sqrt(1/N),size=(N,M))))
        self.bias   = torch.nn.Parameter(torch.zeros((1,M)))
        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        # TODO:
        stacked     = torch.cat((x,state_old),dim=1)
        reset_gate  = F.sigmoid(torch.matmul(stacked,self.weight_r.float()) + self.bias_r.float())
        update_gate = F.sigmoid(torch.matmul(stacked,self.weight_u.float()) + self.bias_u.float())
        
        stacked1    = torch.cat((x,(torch.mul(reset_gate,state_old))),dim=1)
        cand_cell   = torch.tanh(torch.matmul(stacked1,self.weight.float()) + self.bias.float())     # Matmul = dot
        
        state_new   = torch.mul(update_gate,state_old) + torch.mul((1-update_gate),cand_cell)        # Mul = element-wise multi
        return state_new

######################################################################################################################
class RNNCell(nn.Module):
    def __init__(self,hidden_state_sizes,input_size):
        ''' Args:
                hidden_state_size: Integer defining the size of the hidden state of rnn cell
                inputSize: Integer defining the number of input features to the rnn
            Returns:
                self.weight: A nn.Parameter with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                             variance scaling with zero mean.
                self.bias:   A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.    
        '''
        super(RNNCell, self).__init__()
        N                       = hidden_state_sizes+input_size
        M                       = hidden_state_sizes
        
        self.weight = torch.nn.Parameter(torch.from_numpy(np.random.normal(0, np.sqrt(1/N),size=(N,M))))
        self.bias   = torch.nn.Parameter(torch.zeros((1,M)))

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        stacked     = torch.cat((x,state_old),dim=1)
        state_new   = torch.tanh(torch.matmul(stacked,self.weight.float()) + self.bias)
        
        return state_new

######################################################################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words exsisting 
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 0.0000000001                                      # Used to not divide on zero
    logit = torch.transpose(logits,1,2)                     # Correct dimensions
    loss = F.cross_entropy(logit,yTokens,reduction='none')  # Total loss
    relevant_loss = loss*yWeights                           # Loss for non-empty words
    sumLoss  = relevant_loss.sum()                          # Sum of all non-empty words
    meanLoss = (relevant_loss / (yWeights.sum()+eps)).sum() # Mean of all non-empty words

    return sumLoss, meanLoss