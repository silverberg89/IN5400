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

"""Implementation of convolution forward and backward pass"""

import numpy as np

def conv_layer_forward(input_layer, weight, bias, pad_size=1, stride=1):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of M data points, each with C channels, height H and
    width W. We convolve each input with C_o different filters, where each filter
    spans all C_i channels and has height H_w and width W_w.

    Args:
        input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
    """
    # Draw batch size, number of filters and channels
    batch_size  = input_layer.shape[0]
    num_filters = weight.shape[0]
    channels_x  = input_layer.shape[1]
    channels_w  = weight.shape[1]
    
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    
    # Define dimensions of input image and applied filter
    filter_height = weight.shape[-2]
    filter_width  = weight.shape[-1]
    input_height  = input_layer.shape[-2]
    input_width   = input_layer.shape[-1]
    
    # Calculate output layer dimensionality
    output_height = int(1+(input_height-filter_height+2*pad_size)/stride)
    output_width  = int(1+(input_width-filter_width+2*pad_size)/stride)
    
    # Create output holder
    output_layer = np.zeros(shape=(batch_size,num_filters,output_height,output_width))
    
    # Introduce padding for input layer
    pad_width_per_dim = [[0],[0],[pad_size],[pad_size]] # Input_layer: [[dim1],[dim2],[dim3],[dim4]]
    input_padded      = np.pad(input_layer,pad_width_per_dim,mode="constant",constant_values=0)
    
    # Construct movement of window
    for i in range(output_height):
        for j in range(output_width):
            # Set window domain
            i_from = i*stride
            i_to   = i*stride + filter_height
            j_from = j*stride
            j_to   = j*stride + filter_width
            for c in range(output_layer.shape[1]):
                # Perform convolution operation
                term1_o = input_padded[:,:,i_from:i_to,j_from:j_to]
                term2_o = weight[c,:,:,:]
                operation = np.sum(np.multiply(term1_o,term2_o),axis=(1, 2, 3))
                # Complete output layer with bias as: (batch_size, num_filters, height_y, width_y) = [:,c,i,j]
                output_layer[:,c,i,j] = operation + bias[c]

    return output_layer

def conv_layer_backward(output_layer_gradient, input_layer, weight, bias, pad_size, stride):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Args:
        output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
            (batch_size, num_filters, height_y, width_y)
        input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
        weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
        bias: Biases of shape (num_filters)

    Returns:
        input_layer_gradient: Gradient of the loss L with respect to the input layer x
        weight_gradient: Gradient of the loss L with respect to the filters w
        bias_gradient: Gradient of the loss L with respect to the biases b
    """
    # Create holders with same shape as input data
    input_layer_gradient = np.zeros(input_layer.shape[:])
    weight_gradient      = np.zeros(weight.shape[:])
    bias_gradient        = np.zeros(weight.shape[0])
    
    # Draw batch size, number of filters, channels and dimensions
    batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
    batch_size, channels_x, height_x, width_x = input_layer.shape
    num_filters, channels_w, height_w, width_w = weight.shape
    
    # Clarify parameters
    output_height   = height_y
    output_weight   = width_y
    output_channels = channels_y
    filter_height   = height_w
    filter_width    = width_w
    
    # Test necessary constraints
    assert num_filters == channels_y, (
        "The number of filters must be the same as the number of output layer channels")
    assert channels_w == channels_x, (
        "The number of filter channels be the same as the number of input layer channels")
    
    # Introduce padded holders
    pad_width_per_dim = [[0],[0],[pad_size],[pad_size]] # Input_layer: [[dim1],[dim2],[dim3],[dim4]]
    input_padded      = np.pad(input_layer,pad_width_per_dim,mode="constant",constant_values=0)
    input_padded_grad = np.zeros(input_padded.shape[:])
    
    # Backprop iteration
    for i in range(output_height):
        for j in range(output_weight):
            # Set window domain
            i_from = i*stride
            i_to   = i*stride + filter_height
            j_from = j*stride
            j_to   = j*stride + filter_width
            for s in range(batch_size):
                # Perform calculation of input grads
                term1_i = weight
                term2_i = (output_layer_gradient[s,:,i,j])[:,np.newaxis,np.newaxis,np.newaxis]
                input_padded_grad[s,:,i_from:i_to,j_from:j_to] += np.sum(np.multiply(term1_i,term2_i),axis=0) # Sum by first axis
            for c in range(output_channels):
                # Perform calculation of weight grads
                term1_w = input_padded[:,:,i_from:i_to,j_from:j_to]
                term2_w = (output_layer_gradient[:,c,i,j])[:,np.newaxis,np.newaxis,np.newaxis]
                weight_gradient[c,:,:,:] += np.sum(np.multiply(term1_w,term2_w),axis=0) # Sum by first axis
    
    # Draw grads from padded domain
    input_layer_gradient = input_padded_grad[:,:,pad_size:-pad_size,pad_size:-pad_size]
    
    # Sum output layer grads to get bias grads
    bias_gradient = np.sum(output_layer_gradient,axis=(0, 2, 3))
    
    return input_layer_gradient, weight_gradient, bias_gradient

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad