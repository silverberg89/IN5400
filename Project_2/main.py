#Import modules
import torch
import numpy as np
from sourceFiles import cocoSource
from unit_tests import unit_tests

import warnings
warnings.filterwarnings("ignore")

# Set seed
torch.manual_seed(99)
np.random.seed(99)

#----TASK 1a------------------------------------------------------------- [Works]
''' Vanilla recurrent neural network (RNN) '''
#Defining dummy variables
hidden_state_sizes = 512
inputSize  = 256
batch_size = 128
x          = torch.Tensor(torch.rand(batch_size, inputSize))
state_old  = torch.Tensor(torch.rand(batch_size, hidden_state_sizes))

# You should implement this function
cell        = cocoSource.RNNCell(hidden_state_sizes, inputSize)
cell.forward(x, state_old)

#Check implementation
unit_tests.RNNcell_test()

#----TASK 1b------------------------------------------------------------- [Works]
''' Gated recurrent units (GRU) '''
#Defining dummy variables
hidden_state_sizes = 512
inputSize  = 648
batch_size = 128
x          = torch.tensor(torch.rand(batch_size, inputSize))
state_old  = torch.tensor(torch.rand(batch_size, hidden_state_sizes))

# You should implement this function
cell   = cocoSource.GRUCell(hidden_state_sizes, inputSize)
cell.forward(x, state_old)

#Check implementation
unit_tests.GRUcell_test()

#----TASK 1c------------------------------------------------------------- [Works]
''' Loss function '''
weight_dir = 'unit_tests/loss_fn_tensors.pt'
checkpoint = torch.load(weight_dir)
    
logits     = checkpoint['logits']
yTokens    = checkpoint['yTokens']
yWeights   = checkpoint['yWeights']

sumLoss, meanLoss = cocoSource.loss_fn(logits, yTokens, yWeights)

#Check implementation
unit_tests.loss_fn_test()

#----TASK 1d------------------------------------------------------------- [Works]
''' RNN train '''
#Config
my_dir = 'unit_tests/RNN_tensors_is_train_True.pt'
checkpoint = torch.load(my_dir)

outputLayer   = checkpoint['outputLayer']
Embedding     = checkpoint['Embedding']
xTokens       = checkpoint['xTokens']
initial_hidden_state  = checkpoint['initial_hidden_state']
input_size            = checkpoint['input_size']
hidden_state_size     = checkpoint['hidden_state_size']
num_rnn_layers        = checkpoint['num_rnn_layers']
cell_type             = checkpoint['cell_type']
is_train              = checkpoint['is_train']

# You should implement this function
myRNN = cocoSource.RNN(input_size, hidden_state_size, num_rnn_layers, cell_type)
logits, current_state = myRNN(xTokens, initial_hidden_state, outputLayer, Embedding, is_train)

#Check implementation
is_train = True
unit_tests.RNN_test(is_train)

is_train = False
unit_tests.RNN_test(is_train)

#----TASK 1e------------------------------------------------------------- [Works]
''' imageCaption '''
#Config
my_dir = 'unit_tests/imageCaptionModel_tensors.pt'
checkpoint = torch.load(my_dir)
config                            = checkpoint['config']
vgg_fc7_features                  = checkpoint['vgg_fc7_features']
xTokens                           = checkpoint['xTokens']
is_train                          = checkpoint['is_train']
myImageCaptionModelRef_state_dict = checkpoint['myImageCaptionModelRef_state_dict']
logitsRef                         = checkpoint['logitsRef']
current_hidden_state_Ref          = checkpoint['current_hidden_state_Ref']


myImageCaptionModel = cocoSource.imageCaptionModel(config)
logits, current_hidden_state = myImageCaptionModel(vgg_fc7_features, xTokens,  is_train)

#Check implementation
unit_tests.imageCaptionModel_test()

#-------------------------------------------------------------------------- [Do not work]
from utils.dataLoader import DataLoaderWrapper
from utils.saverRestorer import SaverRestorer
from utils.model import Model
from utils.trainer import Trainer
from utils.validate import plotImagesAndCaptions

#Path if you work on personal computer
data_dir = 'data/coco/'

#Path if you work on UIO IFI computer
#data_dir = '/projects/in5400/oblig2/coco/'

#Path if you work on one of the ML servers
#data_dir = '/shared/in5400/coco/'

#train
modelParam = {
        'batch_size': 128,          # Training batch size
        'cuda': {'use_cuda': True,  # Use_cuda=True: use GPU
                 'device_idx': 0},  # Select gpu index: 0,1,2,3
        'numbOfCPUThreadsUsed': 3,  # Number of cpu threads use in the dataloader
        'numbOfEpochs': 1,         # Number of epochs
        'data_dir': data_dir,       # data directory
        'img_dir': 'loss_images/',
        'modelsDir': 'storedModels/',
        'modelName': 'model_0/',    # name of your trained model
        'restoreModelLast': 0,
        'restoreModelBest': 0,
        'modeSetups':   [['train', True], ['val', True]],
        'inNotebook': True,         # If running script in jupyter notebook
        'inference': False
}

config = {
        'optimizer': 'adam',             # 'SGD' | 'adam' | 'RMSprop'
        'learningRate': {'lr': 0.0005},  # learning rate to the optimizer
        'weight_decay': 0,               # weight_decay value
        'VggFc7Size': 4096,              # Fixed, do not change
        'embedding_size': 128,           # word embedding size
        'vocabulary_size': 4000,        # number of different words
        'truncated_backprop_length': 20,
        'hidden_state_sizes': 256,       #
        'num_rnn_layers': 2,             # number of stacked rnn's
        'cellType': 'RNN'                # RNN or GRU
        }


# create an instance of the model you want
model = Model(config, modelParam)

# create an instacne of the saver and resoterer class
saveRestorer = SaverRestorer(config, modelParam)
model        = saveRestorer.restore(model)

# create your data generator
dataLoader = DataLoaderWrapper(config, modelParam)

# here you train your model
trainer = Trainer(model, modelParam, config, dataLoader, saveRestorer)
trainer.train()