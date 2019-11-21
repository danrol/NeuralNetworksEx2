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


def gradientDescent(x, y, k, w1, w2, b1, b2):
    lr = 0.09
    m = 4
    costs = []
    for i in range(max_num_of_epocs):
        #forward propagation
        a1, z1, a2, z2 = forward_propagation(x, k, w1, w2, b1, b2)

        #backward propagation
        delta2, Delta1, Delta2 = back_propogation(a1, a2, x, z1, z2, y)

        w1 -= lr*(1/m)*Delta1
        w2 -= lr*(1/m)*Delta2

        c = np.mean(np.abs(delta2))
        costs.append(c)

        if i % 1000 ==0:
            print(f"Iteration: {i} Error: {c}")

    print("Training complete.")
    z3 = forward_propagation(x, y, k, w1, w2, b1, b2)

    plt.plot(costs)
    plt.show()



def sigmoid_derivative(x):
    return tf.sigmoid(x)*(1-tf.sigmoid(x))

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

def back_propogation(a1, a2, z0, z1, z2, y, w2):
    delta2 = z2 - y
    Delta2 = tf.matmul(tf.transpose(z1), delta2)
    delta1 = (tf.transpose(tf.tensordot(w2[1:,:])))*sigmoid_derivative(a1)
    Delta1 = tf.matmul(z0.T, delta1)
    return delta2, Delta1, Delta2

def xor_neural_network(data, excpected_data, k,  learning_rate):
    amount_input_neurons = 2
    amount_output_neurons = 1

    # define placeholder that will be used later after tensorflow session starts
    x = tf.compat.v1.placeholder(tf.float32, [None, amount_input_neurons])
    y = tf.compat.v1.placeholder(tf.float32, [None, amount_output_neurons])

    w1 = tf.Variable(tf.random.uniform([amount_input_neurons, k], minval=-1, maxval=1, seed=0),
                                       dtype=tf.dtypes.float32, name=None)
    w2 = tf.Variable(tf.random.uniform([k, 1], minval=-1, maxval=1, seed=0),dtype=tf.dtypes.float32,  name=None)
    b1 = tf.compat.v1.Variable(tf.random.uniform([1, k], minval=-1, maxval=1, seed=0), dtype=tf.dtypes.float32, name=None)
    b2 = tf.compat.v1.Variable(tf.random.uniform([1, 1], minval=-1, maxval=1, seed=0), dtype=tf.dtypes.float32, name=None)

    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    #squared = tf.square(final_output - y)
    #mse_loss = tf.reduce_sum(squared)

if __name__ == '__main__':
    input_data_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # expected results from activating XOR (corresponds to the lists in input_train list)
    expected_input_results = np.array([[0], [1], [1], [0]])
    data_validation = np.array([[1, 0.1], [1, 0.9], [0.9, 0.9], [0.1, 0.9]])
    expected_data_validation_results = np.array([[1], [0], [0], [1]])
    xor_neural_network(input_data_x, expected_input_results, 4, 0.09)
