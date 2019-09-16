#this model is for the IV17 Baidu Reactive model  paper
#by Yuanwei 
#Email: weiy1991@sjtu.edu.cn
#Date: 2018-01-16

import tensorflow as tf
import numpy as np
import scipy

# functions for the layer
def weight_var(shape):
	initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
	return tf.Variable(initial)

def bias_var(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W, stride):
	return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

# end functions for the layer

# the tensor for input
x = tf.placeholder(tf.float32, shape=[None, 320, 320, 3 ])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

x_image = x
# end the tensor for input

# model structure

## convolutional layer

### first layer
W_conv1 = weight_var([5, 5, 3, 24])
b_conv1 = bias_var([24])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 4) + b_conv1)
### end first layer

### second layer
W_conv2 = weight_var([5, 5, 24, 48])
b_conv2 = bias_var([48])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
### end second layer

### third layer
W_conv3 = weight_var([5, 5, 48, 64])
b_conv3 = bias_var([64])

h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
### end third layer

### forth layer
W_conv4 = weight_var([3, 3, 64, 96])
b_conv4 = bias_var([96])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 2) + b_conv4)
### end forth layer


### fifth layer
W_conv5 = weight_var([3, 3, 96, 128])
b_conv5 = bias_var([128])
h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 2) + b_conv5)
### end fifth layer


###64@7*17

## end convolutional layer

## flatten

h_flatten = tf.nn.relu(tf.contrib.layers.flatten(h_conv5))

## end flatten

## dese1
keep_prob = tf.placeholder(tf.float32)

W_fc1 = weight_var([1152 ,512])
b_fc1 = bias_var([512])

h_fc1 = tf.nn.relu(tf.matmul(h_flatten, W_fc1) + b_fc1)

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
## end dese1

## output
W_fc2 = weight_var([512 ,1])
b_fc2 = bias_var([1])

#y = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y = tf.multiply(tf.atan(tf.matmul(h_fc1_drop, W_fc2) + b_fc2), 2)
## end output

# end model structure




