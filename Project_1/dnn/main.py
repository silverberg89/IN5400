#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#
#                                                                               #
# Part of mandatory assignment 1 in                                             #
# IN5400 - Machine Learning for Image analysis                                  #
# University of Oslo                                                            #
#                                                                               #
#                                                                               #
# Ole-Johan Skrede    olejohas at ifi dot uio dot no                            #
# 2019.02.12                                                                    #
#                                                                               #
#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::#

"""
Main routine capable of training a dense neural network, and also running inference.

This program builds an L-layer dense neural network. The number of nodes in each layer is set in
the configuration.

By default, every node has a ReLu activation, except the final layer, which has a softmax output.
We use a cross-entropy loss for the cost function, and we use a stochastic gradient descent
optimization routine to minimize the cost function.

Custom configuration for experimentation is possible.
"""

import os
import numpy as np
import import_data
import run
import model
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def config():
    """Return a dict of configuration settings used in the program"""

    conf = {}

    # Determine what dataset to run on. 'mnist', 'cifar10' and 'svhn' are currently supported.
    conf['dataset'] = 'mnist'
    # Relevant datasets will be put in the location data_root_dir/dataset.
    conf['data_root_dir'] = "/tmp/data"

    # Number of input nodes. This is determined by the dataset in runtime.
    conf['input_dimension'] = None
    # Number of hidden layers, with the number of nodes in each layer.
    conf['hidden_dimensions'] = [128, 32]
    # Number of classes. This is determined by the dataset in runtime.
    conf['output_dimension'] = None
    # This will be determined in runtime when input_dimension and output_dimension is set.
    conf['layer_dimensions'] = [784, 128, 32, 10]

    # Size of development partition of the training set
    conf['devel_size'] = 5000
    # What activation function to use in the nodes in the hidden layers.
    conf['activation_function'] = 'relu'
    # The number of steps to run before termination of training. One step is one forward->backward
    # pass of a mini-batch
    conf['max_steps'] = 2000
    # The batch size used in training.
    conf['batch_size'] = 128
    # The step size used by the optimization routine.
    conf['learning_rate'] = 1.0e-2

    # Whether or not to write certain things to stdout.
    conf['verbose'] = False
    # How often (in steps) to log the training progress. Prints to stdout if verbose = True.
    conf['train_progress'] = 10
    # How often (in steps) to evaluate the method on the development partition. Prints to stdout
    # if verbose = True.
    conf['devel_progress'] = 100

    return conf

def plot_progress(train_progress, devel_progress, out_filename=None):
    """Plot a chart of the training progress"""

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    ax1.plot(train_progress['steps'], train_progress['ccr'], 'b', label='Training set ccr')
    ax1.plot(devel_progress['steps'], devel_progress['ccr'], 'r', label='Development set ccr')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Correct classification rate')
    ax1.legend(loc='lower left', bbox_to_anchor=(0.6, 0.52), framealpha=1.0)

    ax2 = ax1.twinx()
    ax2.plot(train_progress['steps'], train_progress['cost'], 'g', label='Training set cost')
    ax2.set_ylabel('Cross entropy cost')
    gl2 = ax2.get_ygridlines()
    for gl in gl2:
        gl.set_linestyle(':')
        gl.set_color('k')

    ax2.legend(loc='lower left', bbox_to_anchor=(0.6, 0.45), framealpha=1.0)
    plt.title('Training progress')
    fig.tight_layout()

    if out_filename is not None:
        plt.savefig(out_filename)

    plt.show()

def get_data(conf):
    """Return data to be used in this session.

    Args:
        conf: Configuration dictionary
    Returns:
        X_train: numpy array of floats with shape [input_dimension, num train examples] in [0, 1].
        Y_train: numpy array of integers with shape [output_dimension, num train examples].
        X_devel: numpy array of floats with shape [input_dimension, num devel examples] in [0, 1].
        Y_devel: numpy array of integers with shape [output_dimension, num devel examples].
        X_test: numpy array of floats with shape [input_dimension, num test examples] in [0, 1].
        Y_test: numpy array of integers with shape [output_dimension, num test examples].
    """

    data_dir = os.path.join(conf['data_root_dir'], conf['dataset'])
    if conf['dataset'] == 'cifar10':
        conf['input_dimension'] = 32*32*3
        conf['output_dimension'] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_cifar10(
            data_dir, conf['devel_size'])
    elif conf['dataset'] == 'mnist':
        conf['input_dimension'] = 28*28*1
        conf['output_dimension'] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_mnist(
            data_dir, conf['devel_size'])
    elif conf['dataset'] == 'svhn':
        conf['input_dimension'] = 32*32*3
        conf['output_dimension'] = 10
        X_train, Y_train, X_devel, Y_devel, X_test, Y_test = import_data.load_svhn(
            data_dir, conf['devel_size'])

    conf['layer_dimensions'] = ([conf['input_dimension']] +
                                conf['hidden_dimensions'] +
                                [conf['output_dimension']])

    if conf['verbose']:
        print("Train dataset:")
        print("  shape = {}, data type = {}, min val = {}, max val = {}".format(X_train.shape,
                                                                                X_train.dtype,
                                                                                np.min(X_train),
                                                                                np.max(X_train)))
        print("Development dataset:")
        print("  shape = {}, data type = {}, min val = {}, max val = {}".format(X_devel.shape,
                                                                                X_devel.dtype,
                                                                                np.min(X_devel),
                                                                                np.max(X_devel)))
        print("Test dataset:")
        print("  shape = {}, data type = {}, min val = {}, max val = {}".format(X_test.shape,
                                                                                X_test.dtype,
                                                                                np.min(X_test),
                                                                                np.max(X_test)))

    return X_train, Y_train, X_devel, Y_devel, X_test, Y_test

def get_batch_indices(indices, start_index, end_index):
    """Return the indices of the examples that are to form a batch.

    This is done so that if end_index > len(example_indices), we will include the remainding
    indices, in addition to the first indices in the example_indices list.

    Args:
        indices: 1D numpy array of integers
        start_index: integer > 0 and smaller than len(example_indices)
        end_index: integer > start_index
    Returns:
        1D numpy array of integers
    """
    n = len(indices)
    return np.hstack((indices[start_index:min(n, end_index)], indices[0:max(end_index-n, 0)]))

def evaluate(conf, params, X_data, Y_data):
    """Evaluate a trained model on X_data.

    Args:
        conf: Configuration dictionary
        params: Dictionary with parameters
        X_data: numpy array of floats with shape [input dimension, number of examples]
        Y_data: numpy array of integers with shape [output dimension, number of examples]
    Returns:
        num_correct_total: Integer
        num_examples_evaluated: Integer
    """

    num_examples = X_data.shape[1]
    num_examples_evaluated = 0
    num_correct_total = 0
    start_ind = 0
    end_ind = conf['batch_size']
    while True:
        X_batch = X_data[:, start_ind: end_ind]
        Y_batch = model.one_hot(Y_data[start_ind: end_ind], conf['output_dimension'])
        Y_proposal, _ = model.forward(conf, X_batch, params, is_training=False)
        _, num_correct = model.cross_entropy_cost(Y_proposal, Y_batch)
        num_correct_total += num_correct

        num_examples_evaluated += end_ind - start_ind

        start_ind += conf['batch_size']
        end_ind += conf['batch_size']

        if end_ind >= num_examples:
            end_ind = num_examples

        if start_ind >= num_examples:
            break

    return num_correct_total, num_examples_evaluated

def main_test():
    print("----------START OF TESTS-----------")
    # Get configuration
    conf = config()
    
    ################################### Task 1.1: Parameter initialization
    
    from model import initialization
    params = initialization(conf)
    
    ################################### Task 1.2: Forward propagation
    
    # Import Activation functions [1.2a & 1.2b]
    from model import activation
    from model import softmax
    
    # Test Activation functions
    from tests import task_2a
    from tests import task_2b
    input_Z, expected_A = task_2a()
    A = activation(input_Z, 'relu')
    print('Activation valid?:',np.array_equal(expected_A, A))
    input_Z, expected_S = task_2b()
    S = softmax(input_Z)
    print('Softmax valid?:',np.array_equal(np.round(expected_S,decimals=3), np.round(S,decimals=3)))
    
    # Import Forward propagation [1.2c]
    from model import forward
    from tests import task_2c
    
    ### Test Forward propagation
    conf, X_batch, params, expected_Z_1, expected_A_1, expected_Z_2, expected_Y_proposed = task_2c()
    Y_proposed, features = forward(conf, X_batch, params, is_training=True)
    print('feature Z_1 valid?:',np.array_equal(expected_Z_1, np.round(features['Z_1'],decimals=8)))
    print('feature A_1 valid?:',np.array_equal(expected_A_1, np.round(features['A_1'],decimals=8)))
    print('feature Z_2 valid?:',np.array_equal(expected_Z_2, np.round(features['Z_2'],decimals=8)))
    print('proposed Y valid?:',np.array_equal(expected_Y_proposed, np.round(Y_proposed,decimals=8)))
    
    ################################### Task 1.3: Cross Entropy cost function
    
    # Import Cost function
    from model import cross_entropy_cost
    from tests import task_3
    
    ### Test Cost function
    Y_proposed, Y_batch, expected_cost_value, expected_num_correct = task_3()
    cost_value, num_correct = cross_entropy_cost(Y_proposed, Y_batch)
    print('Cost value valid?:',np.array_equal(np.round(expected_cost_value,decimals=4), np.round(cost_value,decimals=4)))
    print('Number of succesess valid?:',np.array_equal(expected_num_correct, np.round(num_correct,decimals=4)))
    
    ################################### Task 1.4: Backward propagation
    
    # Import Derivative of the activation function [1.4a]
    from model import activation_derivative
    from tests import task_4a
    
    # Test Derivative of activation
    input_Z, expected_dg_dz = task_4a()
    dg_dz = activation_derivative(input_Z, "relu")
    print('Derivative function valid?:',np.array_equal(expected_dg_dz, np.round(dg_dz,decimals=4)))

    # Import Backward propagation [1.4b]
    from model import backward
    from tests import task_4b
    
    # Test Backward propagation
    (conf, Y_proposed, Y_batch, params, features,
     expected_grad_W_1, expected_grad_b_1, expected_grad_W_2, expected_grad_b_2) = task_4b()
    grad_params = backward(conf, Y_proposed, Y_batch, params, features)
    print('Grad_W_1 valid?:',np.array_equal(np.round(expected_grad_W_1,decimals=4), np.round(grad_params["grad_W_1"],decimals=4)))
    print('Grad_b_1 valid?:',np.array_equal(np.round(expected_grad_b_1,decimals=4), np.round(grad_params["grad_b_1"][:, np.newaxis],decimals=4)))
    print('Grad_W_2 valid?:',np.array_equal(np.round(expected_grad_W_2,decimals=4), np.round(grad_params["grad_W_2"],decimals=4)))
    print('Grad_b_2 valid?:',np.array_equal(np.round(expected_grad_b_2,decimals=4), np.round(grad_params["grad_b_2"][:, np.newaxis],decimals=4)))
    
    ################################### Task 1.5: Update parameters
    
    # Import Update
    from model import gradient_descent_update
    from tests import task_5
    
    # Test Update
    (conf, params, grad_params,
     expected_updated_W_1, expected_updated_b_1, expected_updated_W_2, expected_updated_b_2) = task_5()
    updated_params = gradient_descent_update(conf, params, grad_params)
    
    print('update of W_1 valid?:',np.array_equal(np.round(expected_updated_W_1,decimals=4), np.round(updated_params["W_1"],decimals=4)))
    print('update of b_1 valid?:',np.array_equal(np.round(expected_updated_b_1,decimals=4), np.round(updated_params["b_1"],decimals=4)))
    print('update of W_2 valid?:',np.array_equal(np.round(expected_updated_W_2,decimals=4), np.round(updated_params["W_2"],decimals=4)))
    print('update of b_2 valid?:',np.array_equal(np.round(expected_updated_b_2,decimals=4), np.round(updated_params["b_2"],decimals=4)))

    print("----------END OF TESTS-----------")
    
def main_exceed():
    """Run the program according to specified configurations."""
    ################################### Task 1.6b: Exceed results
    conf = config()
    conf['dataset'] = 'mnist'                           # Dataset
    conf['max_steps'] = 5000                            # Training steps
    conf['learning_rate'] = 1.0e-2*0.5                  # Learning rate
    conf['hidden_dimensions'] = [128, 64, 32]           # Hidden layers & Nodes
    conf['batch_size'] = 64                             # Batch size
    conf['activation_function'] = 'sigmoid'             # Hidden activation function
    print("----------START DNN ON: ",conf['dataset'])

    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)

    params, train_progress, devel_progress = run.train(conf, X_train, Y_train, X_devel, Y_devel)

    plot_progress(train_progress, devel_progress)

    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_test, Y_test)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("----------END DNN ON: ",conf['dataset'])
    
    ################################### Task 1.6a: Reproduce results cifar10
    conf = config()
    conf['dataset'] = 'cifar10'                         # Dataset
    conf['max_steps'] = 12000                           # Training steps
    conf['learning_rate'] = 1.0e-2*4                    # Learning rate
    conf['hidden_dimensions'] = [256, 128, 64, 32]      # Hidden layers & Nodes
    conf['batch_size'] = 32                             # Batch size
    conf['activation_function'] = 'tanh'                # Hidden activation function

    print("----------START DNN ON: ",conf['dataset'])

    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)

    params, train_progress, devel_progress = run.train(conf, X_train, Y_train, X_devel, Y_devel)

    plot_progress(train_progress, devel_progress)

    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_test, Y_test)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("----------END DNN ON: ",conf['dataset'])
    
    ################################### Task 1.6a: Reproduce results svhn
    conf = config()
    conf['dataset'] = 'svhn'                            # Dataset
    conf['max_steps'] = 12000                           # Training steps
    conf['learning_rate'] = 1.0e-2*0.5                  # Learning rate
    conf['hidden_dimensions'] = [256, 64, 32]           # Hidden layers & Nodes
    conf['batch_size'] = 64                             # Batch size
    conf['activation_function'] = 'sigmoid'             # Hidden activation function
    print("----------START DNN ON: ",conf['dataset'])

    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)

    params, train_progress, devel_progress = run.train(conf, X_train, Y_train, X_devel, Y_devel)

    plot_progress(train_progress, devel_progress)

    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_test, Y_test)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("----------END DNN ON: ",conf['dataset'])
    
def main():
    """Run the program according to specified configurations."""
    ################################### Task 1.6a: Reproduce results mnist
    conf = config()
    conf['dataset'] = 'mnist'
    print("----------START DNN ON: ",conf['dataset'])

    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)

    params, train_progress, devel_progress = run.train(conf, X_train, Y_train, X_devel, Y_devel)

    plot_progress(train_progress, devel_progress)

    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_test, Y_test)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("----------END DNN ON: ",conf['dataset'])
    
    ################################### Task 1.6a: Reproduce results cifar10
    conf = config()
    conf['dataset'] = 'cifar10'
    conf['max_steps'] = 10000
    print("----------START DNN ON: ",conf['dataset'])

    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)

    params, train_progress, devel_progress = run.train(conf, X_train, Y_train, X_devel, Y_devel)

    plot_progress(train_progress, devel_progress)

    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_test, Y_test)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("----------END DNN ON: ",conf['dataset'])
    
    ################################### Task 1.6a: Reproduce results svhn
    conf = config()
    conf['dataset'] = 'svhn'
    conf['max_steps'] = 10000
    print("----------START DNN ON: ",conf['dataset'])

    X_train, Y_train, X_devel, Y_devel, X_test, Y_test = get_data(conf)

    params, train_progress, devel_progress = run.train(conf, X_train, Y_train, X_devel, Y_devel)

    plot_progress(train_progress, devel_progress)

    print("Evaluating train set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_train, Y_train)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating development set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_devel, Y_devel)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("Evaluating test set")
    num_correct, num_evaluated = run.evaluate(conf, params, X_test, Y_test)
    print("CCR = {0:>5} / {1:>5} = {2:>6.4f}".format(num_correct, num_evaluated,
                                                     num_correct/num_evaluated))
    print("----------END DNN ON: ",conf['dataset'])
    
if __name__ == "__main__":
    main_test()     # Task [1.1 -> 1.5]    Benchmark against tests
    main()          # Task 1.6a            Run datasets for reproduction
    main_exceed()   # Task 1.6b            Run datasets for improvement