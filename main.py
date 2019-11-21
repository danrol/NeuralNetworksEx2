#EX2 XOR using Gradient Descent and Cross Entropy loss
#1 1 --> 0
#1 0 --> 1
#0 1 --> 1
#0 0 --> 0
#Daniil Rolnik
#334018009
#k - amount of hidden neurons in hidden layer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
max_num_of_epocs = 40000
temperature = 0.001


def gradientDescent():
    for i in range(max_num_of_epocs):
        pass

def forward_propagation(x, k, w1, w2, b1, b2, ):
    a1 = tf.matmul(x, w1) + b1
    z1 = tf.sigmoid(a1 / temperature)

    if k == 1:
        conc_hidden_layer_res = tf.concat([z1, x], 1)
    else:
        conc_hidden_layer_res = z1

    a2 = tf.matmul(conc_hidden_layer_res, w2) + b2
    z2 = tf.sigmoid(a2 / temperature)

    return a1, z1, a2, z2

def xorNeuralNetwork(data, excpected_data, k, bridge, learning_rate):
    amount_input_neurons = 2
    amount_output_neurons = 1

    # define placeholder that will be used later after tensorflow session starts
    x = tf.compat.v1.placeholder(tf.float32, [None, amount_input_neurons])
    y = tf.compat.v1.placeholder(tf.float32, [None, amount_output_neurons])

    w1 = tf.Variable(tf.random_uniform([amount_input_neurons, k], minval=-1, maxval=1, seed=0),
                                       dtype=tf.dtypes.float32, name=None)
    w2 = tf.Variable(tf.random_uniform([k, 1], minval=-1, maxval=1, seed=0),dtype=tf.dtypes.float32,  name=None)
    b1 = tf.compat.v1.Variable(tf.random_uniform([1, k], minval=-1, maxval=1, seed=0), dtype=tf.dtypes.float32, name=None)
    b2 = tf.compat.v1.Variable(tf.random_uniform([1, 1], minval=-1, maxval=1, seed=0), dtype=tf.dtypes.float32, name=None)

    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    squared = tf.square(final_output - y)
    mse_loss = tf.reduce_sum(squared)

if __name__ == '__main__':
    input_data_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # expected results from activating XOR (corresponds to the lists in input_train list)
    expected_input_results = [[0], [1], [1], [0]]
    data_validation = [[1, 0.1], [1, 0.9], [0.9, 0.9], [0.1, 0.9]]
    expected_data_validation_results = [[1], [0], [0], [1]]

