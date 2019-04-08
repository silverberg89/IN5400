import numpy as np

################################# Task 2.1: Convolution --- basic forward pass
from conv_layers import conv_layer_forward

batch_size = 1
num_filters = 2

channels_x, height_x, width_x = 3, 4, 4
height_w, width_w = 3, 3

stride = 1
pad_size = 1

x_shape = (batch_size, channels_x, height_x, width_x)
w_shape = (num_filters, channels_x, height_w, width_w)

input_layer = np.linspace(-0.4, 0.3, num=np.prod(x_shape)).reshape(x_shape)
weight = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
bias = np.linspace(-0.1, 0.2, num=num_filters)

output_layer = conv_layer_forward(input_layer, weight, bias, pad_size, stride)

correct_out = np.array(
    [[[[ 0.15470494,  0.28520674,  0.26826174,  0.14451626],   # y[0, 0, 0, :]
       [ 0.28745885,  0.47927338,  0.44816540,  0.25953031],   # y[0, 0, 1, :]
       [ 0.20956242,  0.35484143,  0.32373344,  0.17151746],   # y[0, 0, 2, :]
       [ 0.07288238,  0.14856283,  0.12403051,  0.03908872]],  # y[0, 0, 3, :]

      [[ 0.07425532,  0.04867523,  0.10001606,  0.15511441],   # y[0, 1, 0, :]
       [ 0.15335608,  0.17933360,  0.25065436,  0.26199920],   # y[0, 1, 1, :]
       [ 0.34860297,  0.46461662,  0.53593737,  0.44712967],   # y[0, 1, 2, :]
       [ 0.35662385,  0.45831794,  0.50207146,  0.41387796]]]] # y[0, 1, 3, :]
)
print('Output_layer valid?:',np.array_equal(np.round(output_layer,decimals=8), np.round(correct_out,decimals=8)))

################################# Task 2.1: Convolution --- basic forward pass [MULTI]

from conv_layers import conv_layer_forward

batch_size = 2
num_filters = 2

channels_x, height_x, width_x = 3, 5, 5
height_w, width_w = 3, 3

stride = 2
pad_size = 1

x_shape = (batch_size, channels_x, height_x, width_x)
w_shape = (num_filters, channels_x, height_w, width_w)

input_layer = np.linspace(-0.4, 0.3, num=np.prod(x_shape)).reshape(x_shape)
weight = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
bias = np.linspace(-0.1, 0.2, num=num_filters)

output_layer = conv_layer_forward(input_layer, weight, bias, pad_size, stride)

correct_out = np.array(
    [[[[ 0.17033051,  0.32060403,  0.18923389],   # y[0, 0, 0, :]
       [ 0.33279093,  0.56466886,  0.35157275],   # y[0, 0, 1, :]
       [ 0.18810941,  0.33769913,  0.19424845]],  # y[0, 0, 2, :]

      [[-0.35023427, -0.57793339, -0.28825123],   # y[0, 1, 0, :]
       [-0.43650753, -0.69081423, -0.35310624],   # y[0, 1, 1, :]
       [-0.11705711, -0.23774091, -0.06783842]]], # y[0, 1, 2, :]

     
     [[[-0.07697860, -0.08027605, -0.09796378],   # y[1, 0, 0, :]
       [-0.12792200, -0.17127517, -0.16897303],   # y[1, 0, 1, :]
       [-0.17886539, -0.24267950, -0.21261492]],  # y[1, 0, 2, :]

      [[ 0.47944789,  0.63667342,  0.50154236],   # y[1, 1, 0, :]
       [ 0.71826643,  0.99647208,  0.74183487],   # y[1, 1, 1, :]
       [ 0.59295935,  0.79736735,  0.60228948]]]] # y[1, 1, 2, :]
)

# Compare your output to ours
print('Output_layer valid?:',np.array_equal(np.round(output_layer,decimals=8), np.round(correct_out,decimals=8)))

################################# Task 2.2: Convolution --- basic backward pass
from conv_layers import conv_layer_forward, conv_layer_backward, eval_numerical_gradient_array

np.random.seed(231)

batch_size = 1
num_filters = 2

channels_x, height_x, width_x = 3, 7, 7
height_w, width_w = 3, 3

stride = 1
pad_size = 1

input_layer = np.random.randn(batch_size, channels_x, height_x, width_x)
weight = np.random.randn(num_filters, channels_x, height_w, width_w)
bias = np.random.randn(num_filters,)
output_layer_gradient = np.random.randn(batch_size, num_filters, height_x, width_x)

numeric_input_layer_gradient = eval_numerical_gradient_array(
    lambda x: conv_layer_forward(x, weight, bias, pad_size, stride), input_layer, output_layer_gradient)
numeric_weight_gradient = eval_numerical_gradient_array(
    lambda w: conv_layer_forward(input_layer, w, bias, pad_size, stride), weight, output_layer_gradient)
numeric_bias_gradient = eval_numerical_gradient_array(
    lambda b: conv_layer_forward(input_layer, weight, b, pad_size, stride), bias, output_layer_gradient)

input_layer_gradient, weight_gradient, bias_gradient = conv_layer_backward(
    output_layer_gradient, input_layer, weight, bias, pad_size, stride)

# Compare your output to ours
print('gradient of L wrt w, valid?:',np.array_equal(np.round(weight_gradient,decimals=6), np.round(numeric_weight_gradient,decimals=6)))
print('gradient of L wrt x, valid?:',np.array_equal(np.round(input_layer_gradient,decimals=6), np.round(numeric_input_layer_gradient,decimals=6)))
print('gradient of L wrt b, valid?:',np.array_equal(np.round(bias_gradient,decimals=6), np.round(numeric_bias_gradient,decimals=6)))

################################# Task 2.2: Convolution --- basic backward pass [MULTI]
from conv_layers import conv_layer_forward, conv_layer_backward, eval_numerical_gradient_array

np.random.seed(231)

batch_size = 2
num_filters = 2

channels_x, height_x, width_x = 3, 7, 7
height_w, width_w = 3, 3

stride = 1
pad_size = 1

input_layer = np.random.randn(batch_size, channels_x, height_x, width_x)
weight = np.random.randn(num_filters, channels_x, height_w, width_w)
bias = np.random.randn(num_filters,)
output_layer_gradient = np.random.randn(batch_size, num_filters, height_x, width_x)

numeric_input_layer_gradient = eval_numerical_gradient_array(
    lambda x: conv_layer_forward(x, weight, bias, pad_size, stride), input_layer, output_layer_gradient)
numeric_weight_gradient = eval_numerical_gradient_array(
    lambda w: conv_layer_forward(input_layer, w, bias, pad_size, stride), weight, output_layer_gradient)
numeric_bias_gradient = eval_numerical_gradient_array(
    lambda b: conv_layer_forward(input_layer, weight, b, pad_size, stride), bias, output_layer_gradient)

input_layer_gradient, weight_gradient, bias_gradient = conv_layer_backward(
    output_layer_gradient, input_layer, weight, bias, pad_size, stride)

# Compare your output to ours
print('gradient of L wrt w multi, valid?:',np.array_equal(np.round(weight_gradient,decimals=6), np.round(numeric_weight_gradient,decimals=6)))
print('gradient of L wrt x multi, valid?:',np.array_equal(np.round(input_layer_gradient,decimals=6), np.round(numeric_input_layer_gradient,decimals=6)))
print('gradient of L wrt b multi, valid?:',np.array_equal(np.round(bias_gradient,decimals=6), np.round(numeric_bias_gradient,decimals=6)))