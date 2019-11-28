#EX2 XOR using Gradient Descent and Cross Entropy loss
#1 1 --> 0
#1 0 --> 1
#0 1 --> 1
#0 0 --> 0
#Daniil Rolnik
#334018009
#k - amount of hidden neurons in hidden layer
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

max_num_of_epocs = 40000
num_succ_runs = 10
temperature = 0.001
learning_rate = 0.01
huge_num = 1000000
min_change = 0.01

def get_output(x, w1, w2, bridge, b1, b2):
    z1 = tf.matmul(x, w1) + b1
    hidden_layer_res = tf.sigmoid(z1 / temperature)
    if bridge:
        conc_hidden_layer_res = tf.concat([hidden_layer_res, x], 1)
    else:
        conc_hidden_layer_res = tf.concat([hidden_layer_res, x], 1)

    z2 = tf.matmul(conc_hidden_layer_res, w2) + b2

    output = tf.sigmoid(z2 / temperature)

def xor_neural_network(train_data, expected_train_res, val_data, exp_val_res, k, bridge,  learn_rate):
    amount_input_neurons = 2
    amount_output_neurons = 1
    rand_seed = random.seed(350)

    # define placeholder that will be used later after tensorflow session starts
    x = tf.compat.v1.placeholder(tf.float32, [None, amount_input_neurons])
    y = tf.compat.v1.placeholder(tf.float32, [None, amount_output_neurons])
    y = tf.compat.v1.placeholder(tf.float32, [None, 1])

    w1 = tf.Variable(tf.random.uniform([amount_input_neurons, k], minval=-1, maxval=1, seed=0),
                                       dtype=tf.dtypes.float32, name=None)
    w2 = tf.Variable(tf.random.uniform([k, 1], minval=-1, maxval=1, seed=rand_seed),dtype=tf.dtypes.float32,  name=None)
    b1 = tf.compat.v1.Variable(tf.random.uniform([1, k], minval=-1, maxval=1, seed=rand_seed), dtype=tf.dtypes.float32, name=None)
    b2 = tf.compat.v1.Variable(tf.random.uniform([1, 1], minval=-1, maxval=1, seed=rand_seed), dtype=tf.dtypes.float32, name=None)

    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    loss = - tf.reduce_sum((y * tf.log(output)) + (1 - y) * tf.log(1.0 - output))
    train_grad = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss)

    last_loss = huge_num
    success = False
    num_of_succ = 0
    for epoch in range(max_num_of_epocs):
        sess.run(train_grad, {x: train_data, y: expected_train_res})
        val_loss = sess.run(loss, {x: val_data, y: exp_val_res})
        if abs(last_loss-val_loss) < min_change:
            num_of_succ = num_of_succ + 1
            if val_los < min
    final_output = []
    #loss = []
    #result = sess.run([final_output, loss], {x: data, y: expected_data})

    #squared = tf.square(final_output - y)
    #mse_loss = tf.reduce_sum(squared)

if __name__ == '__main__':
    input_data_x = np.array([[ 0, 0],
                             [ 0, 1],
                             [1, 0],
                             [ 1, 1]])
    # expected results from activating XOR (corresponds to the lists in input_train list)
    expected_input_results = np.array([[0], [1], [1], [0]])
    data_validation_input = np.array([[1, 0.1],
                                [1, 0.9],
                                [0.9, 0.9],
                                [0.1, 0.9]])
    expected_data_validation_results = np.array([[1], [0], [0], [1]])
    k = 4
    xor_neural_network(input_data_x, expected_input_results,k , 0.09)
