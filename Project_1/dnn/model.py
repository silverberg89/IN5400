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

"""Define the dense neural network model"""

import numpy as np
from scipy.stats import truncnorm


def one_hot(Y, num_classes):
    """Perform one-hot encoding on input Y.

    It is assumed that Y is a 1D numpy array of length m_b (batch_size) with integer values in
    range [0, num_classes-1]. The encoded matrix Y_tilde will be a [num_classes, m_b] shaped matrix
    with values

                   | 1,  if Y[i] = j
    Y_tilde[i,j] = |
                   | 0,  else
    """
    m = len(Y)
    Y_tilde = np.zeros((num_classes, m))
    Y_tilde[Y, np.arange(m)] = 1
    return Y_tilde


def initialization(conf):
    """Initialize the parameters of the network.

    Args:
        layer_dimensions: A list of length L+1 with the number of nodes in each layer, including
                          the input layer, all hidden layers, and the output layer.
    Returns:
        params: A dictionary with initialized parameters for all parameters (weights and biases) in
                the network.
    """
    # Initilize First layer:
    index = 0
    params = {}
    params["W_" + str(index+1)] = np.random.normal(0, np.sqrt(2/conf["layer_dimensions"][index]),size=(conf['layer_dimensions'][index],conf['layer_dimensions'][index + 1]))
    params["b_" + str(index+1)] = np.zeros(conf['layer_dimensions'][index + 1])
    
    # Initilize Hidden layers:
    hidden_neurons  = conf['layer_dimensions'][1:-1]
    for i in range(0,len(hidden_neurons)-1):
        index +=1
        params["W_" + str(index+1)] = np.random.normal(0, np.sqrt(2/conf["layer_dimensions"][index]),size=(conf['layer_dimensions'][index],conf['layer_dimensions'][index + 1]))
        params["b_" + str(index+1)] = np.zeros(conf['layer_dimensions'][index + 1])
    
    # Initilize Output layer:
    index += 1
    params["W_" + str(index+1)] = np.random.normal(0, np.sqrt(2/conf["layer_dimensions"][index]),size=(conf['layer_dimensions'][index],conf['layer_dimensions'][index + 1]))
    params["b_" + str(index+1)] = np.zeros(conf['layer_dimensions'][index + 1])

    return params


def activation(Z, activation_function):
    """Compute a non-linear activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    if activation_function == 'relu':
        return (np.where(Z>0,Z,0))
    elif activation_function == 'sigmoid':
        return (1 + np.exp(-Z))**(-1)
    elif activation_function == 'tanh':
        return (np.tanh(Z))
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None


def softmax(Z):
    """Compute and return the softmax of the input.

    To improve numerical stability, we do the following

    1: Subtract Z from max(Z) in the exponentials
    2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    z = Z - np.max(Z)                                       # Max trick
    #s = np.exp(z) / np.sum(np.exp(z),axis=1)[:, np.newaxis] # Resulting softmax
    t = z - np.log(np.sum(np.exp(z),axis=0)[np.newaxis, :]) # Log trick
    x = np.exp(t)                                           # Resulting softmax
    return (x)


def forward(conf, X_batch, params, is_training):
    """One forward step.

    Args:
        conf: Configuration dictionary.
        X_batch: float numpy array with shape [n^[0], batch_size]. Input image batch.
        params: python dict with weight and bias parameters for each layer.
        is_training: Boolean to indicate if we are training or not. This function can namely be
                     used for inference only, in which case we do not need to store the features
                     values.

    Returns:
        Y_proposed: float numpy array with shape [n^[L], batch_size]. The output predictions of the
                    network, where n^[L] is the number of prediction classes. For each input i in
                    the batch, Y_proposed[c, i] gives the probability that input i belongs to class
                    c.
        features: Dictionary with
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
               We cache them in order to use them when computing gradients in the backpropagation.
    """
    # Create feature holders
    layers          = len(conf['layer_dimensions'])
    z               = np.zeros(layers-1,dtype=object)
    z_activated     = np.zeros(layers-1,dtype=object)

    # Prop first layer
    index           = 0
    z[0]            = np.dot(X_batch.T,params["W_" + str(index+1)]) + params["b_" + str(index+1)].T
    z_activated[0]  = activation(z[0], 'relu')
    
    # Prop hidden layers
    for i in range(1,layers-2):
        index           += 1
        z[i]            = np.dot(z_activated[i-1],params["W_" + str(index+1)]) + params["b_" + str(index+1)].T
        z_activated[i]  = activation(z[i], 'relu')
        
    # Prop output layer
    index            += 1
    z[-1]            = np.dot(z_activated[-2],params["W_" + str(index+1)]) + params["b_" + str(index+1)].T
    z_activated[-1]  = softmax(z[-1].T).T
    
    # Output as dict
    features = {}
    features["A_" + str(0)] = X_batch
    for j in range(1,layers):
        features["A_" + str(j)] = z_activated[j-1].T
        features["Z_" + str(j)] = z[j-1].T
    Y_proposed = z_activated[-1].T
    
    return Y_proposed, features


def cross_entropy_cost(Y_proposed, Y_batch):
    """Compute the cross entropy cost function.

    Args:
        Y_proposed: numpy array of floats with shape [n_y, m].
        Y_reference: numpy array of floats with shape [n_y, m]. Collection of one-hot encoded
                     true input labels

    Returns:
        cost: Scalar float: 1/m * sum_i^m sum_j^n y_reference_ij log y_proposed_ij
        num_correct: Scalar integer
    """
    # Calculate cost
    cost = -np.sum(np.sum(Y_batch * np.log(Y_proposed))) / len(Y_batch[1])
    
    # Turn Y_proposed into binary matrix
    Y_proposed_binary = np.zeros((np.shape(Y_proposed)))
    for i in range(0,len(Y_proposed[1])):
        Y_proposed_binary[np.argmax(Y_proposed[:,i]),i] = 1
    
    # Test success rate
    Y_batch_temp    = np.where(Y_batch == 0,-1,1)
    num_correct     = np.sum(Y_proposed_binary == Y_batch_temp)
    
    return cost, num_correct


def activation_derivative(Z, activation_function):
    """Compute the gradient of the activation function.

    Args:
        Z: numpy array of floats with shape [n, m]
    Returns:
        numpy array of floats with shape [n, m]
    """
    if activation_function == 'relu':
        return (np.where(Z>=0,1,0))
    elif activation_function == 'sigmoid':
        return (Z * (1 - Z))
    elif activation_function == 'tanh':
        return (1.0 - np.tanh(Z)**2)
    else:
        print("Error: Unimplemented activation function: {}", activation_function)
        return None

def backward(conf, Y_proposed, Y_reference, params, features):
    """Update parameters using backpropagation algorithm.

    Args:
        conf: Configuration dictionary.
        Y_proposed: numpy array of floats with shape [n_y, m].
        features: Dictionary with matrices from the forward propagation. Contains
                - the linear combinations Z^[l] = W^[l]a^[l-1] + b^[l] for l in [1, L].
                - the activations A^[l] = activation(Z^[l]) for l in [1, L].
        params: Dictionary with values of the trainable parameters.
                - the weights W^[l] for l in [1, L].
                - the biases b^[l] for l in [1, L].
    Returns:
        grad_params: Dictionary with matrices that is to be used in the parameter update. Contains
                - the gradient of the weights, grad_W^[l] for l in [1, L].
                - the gradient of the biases grad_b^[l] for l in [1, L].
    """
    # Set up necessary holders and values
    layers = len(conf['layer_dimensions'])
    m      = len(Y_reference[1])
    error  = np.zeros(layers-1,dtype=object)
    grad_w = np.zeros(layers-1,dtype=object)
    grad_b = np.zeros(layers-1,dtype=object)

    # Error & gradients output layer
    error[-1]  = (Y_proposed - Y_reference)
    grad_w[-1] = np.dot(features['A_'+str(layers-2)],error[-1].T) / m
    grad_b[-1] = np.sum(error[-1].T,axis=0)[:,np.newaxis] / m
    
    for i in range(1,layers-1):
        # Error & gradients hidden layers
        der         = activation_derivative(features['Z_'+str(layers-i-1)],'relu')
        dot         = np.dot(error[-(i)].T,params['W_'+str(layers-i)].T).T
        error[-i-1] = dot * der
        # Weights and biases gradients
        grad_w[-i-1] = np.dot(features['A_'+str(layers-2-i)],error[-i-1].T) / m
        grad_b[-i-1] = np.sum(error[-i-1].T,axis=0)[:,np.newaxis] / m
        
    # Output as dict
    grad_params = {}
    for j in range(0,layers-1):
        grad_params["grad_W_" + str(j+1)] = grad_w[j]
        grad_params["grad_b_" + str(j+1)] = np.squeeze(grad_b[j])
        
    return grad_params


def gradient_descent_update(conf, params, grad_params):
    """Update the parameters in params according to the gradient descent update routine.

    Args:
        conf: Configuration dictionary
        params: Parameter dictionary with W and b for all layers
        grad_params: Parameter dictionary with b gradients, and W gradients for all
                     layers.
    Returns:
        params: Updated parameter dictionary.
    """
    layers  = int(len(list(grad_params))/2)
    lr_rate = conf['learning_rate']
    for i in range (1,layers+1):
        # Update weights and biases
        params['W_'+str(i)] -= lr_rate * grad_params['grad_W_'+str(i)]
        params['b_'+str(i)] -= lr_rate * grad_params['grad_b_'+str(i)]

    return params