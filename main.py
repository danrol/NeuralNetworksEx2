#k - amount of hidden neurons in hidden layer
import numpy as np
import tensorflow as tf
max_num_of_epocs = 40000

def gradientDescentXor(data, excpected_data, k, bridge, learning_rate):
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
    sess.run(tf.compat.v1.global_variables_initializer)


if __name__ == '__main__':
    input_data_x = [[0, 0], [0, 1], [1, 0], [1, 1]]
    # expected results from activating XOR (corresponds to the lists in input_train list)
    expected_input_results = [[0], [1], [1], [0]]
    data_validation = [[1, 0.1], [1, 0.9], [0.9, 0.9], [0.1, 0.9]]
    expected_data_validation_results = [[1], [0], [0], [1]]

