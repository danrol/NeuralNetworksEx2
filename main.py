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

max_num_of_steps = 40000
num_succ_runs = 10
temperature = 0.001
#learning_rate = 0.01
min_val_loss = 0.3
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
    return output

def perform_steps(sess, train_grad, x, y, train_data, expected_train_res, loss, val_data, exp_val_res):
    success, count, num_of_succ, last_loss = (False, 0, 0, huge_num)
    for steps in range(max_num_of_steps):
        sess.run(train_grad, {x: train_data, y: expected_train_res})
        val_loss = sess.run(loss, {x: val_data, y: exp_val_res})
        if abs(last_loss - val_loss) < min_change:
            num_of_succ = num_of_succ + 1
            if val_loss < min_val_loss and count >= num_succ_runs:
                success = True
                break
            else:
                count = 0
    return success, steps, val_loss

def run_experiment(text_file, exp_num, train_data, expected_train_res, val_data, exp_val_res, k, bridge, learning_rate):
    steps, succ_count, fail_count, val_loss, train_loss = ([], 0, 0, [], [])
    while succ_count < num_succ_runs:
        result, success = xor_neural_network(train_data, expected_train_res, val_data, exp_val_res, k, bridge, learning_rate)
        b1, b2, loss, w1, w2, steps, val_loss, train_loss = (result[0], result[1], result[2], result[3], result[4], \
            result[5], result[6], result[7])
        if success == True:
            steps.append(steps)
            val_loss.append(val_loss)
            train_loss.append(train_loss)
            succ_count = succ_count + 1
        else:
            fail_count = fail_count + 1
    mean_epochs = np.mean(steps)
    std_epochs = np.std(steps)
    mean_val_loss = np.mean(train_loss)
    std_val_loss_percent = np.std(val_loss)
    mean_train_loss = np.mean(train_loss)
    std_train_loss_percent = np.std(train_loss)
    write_experiment(text_file, exp_num, k, learning_rate, bridge, mean_epochs, std_epochs, fail_count, mean_val_loss,
                     std_val_loss_percent, mean_train_loss, std_train_loss_percent)

def write_experiment(text_file, exp_num, k, learning_rate, bridge, mean_epochs, std_epochs, fail_count,
                     mean_val_loss, std_val_loss_percent, mean_train_loss, std_train_loss_percent):
    result_str = f"experiment {exp_num}:hidden:{k}, LR:{learning_rate}, bridge:{bridge}\n " \
                 f"mean_epocs:{mean_epochs}, std/epocs% {std_epochs},Failures = {fail_count} \n " \
                 f"mean_valid_loss:{mean_val_loss}, stdvalidlossPercent: {std_val_loss_percent}, " \
                 f"meanTrainLoss: {mean_train_loss}, stdTrainLossPercent:{std_train_loss_percent}"
    print(result_str)
    text_file.write(result_str)

def xor_neural_network(train_data, expected_train_res, val_data, exp_val_res, k, bridge, learning_rate):
    amount_input_neurons, amount_output_neurons, rand_seed = (2, 1, 350)

    # define placeholder that will be used later after tensorflow session starts
    x = tf.compat.v1.placeholder(tf.float32, [None, amount_input_neurons])
    y = tf.compat.v1.placeholder(tf.float32, [None, amount_output_neurons])

    w1 = tf.Variable(tf.random.uniform([amount_input_neurons, k], minval=-1, maxval=1, seed=0),
                                       dtype=tf.dtypes.float32, name=None)
    w2 = tf.Variable(tf.random.uniform([k, 1], minval=-1, maxval=1, seed=rand_seed),dtype=tf.dtypes.float32,  name=None)
    b1 = tf.compat.v1.Variable(tf.random.uniform([1, k], minval=-1, maxval=1, seed=rand_seed), dtype=tf.dtypes.float32, name=None)
    b2 = tf.compat.v1.Variable(tf.random.uniform([1, 1], minval=-1, maxval=1, seed=rand_seed), dtype=tf.dtypes.float32, name=None)

    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    output = get_output(x, w1, w2, bridge, b1, b2)
    loss = - tf.reduce_sum((y * tf.log(output)) + (1 - y) * tf.log(1.0 - output))
    train_grad = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    success, steps, val_loss = \
        perform_steps(sess, train_grad, x, y, train_data, expected_train_res, loss, val_data, exp_val_res);

    result_to_return, success = \
        sess.run([b1, b2, loss, w1, w2], {x: val_data, y: expected_data_validation_results})
    train_loss = sess.run(loss, {x: train_data, y: expected_train_res})
    (new_b1, new_b2, new_loss, new_w1, new_w2, steps, val_loss, train_loss) = \
    (result_to_return[0], result_to_return[1], result_to_return[2], result_to_return[3], result_to_return[4], \
     result_to_return[5], result_to_return[6], result_to_return[7])

    train = sess.run(loss, {x: train_data, y: expected_train_res})
    result_to_return = [new_b1, new_b2, new_loss, new_w1, new_w2, steps, val_loss, train_loss]
    return result_to_return, success


def print_inputs(input_data_x, expected_input_results, data_validation_input, expected_data_validation_results):
    print("Input data x:")
    for input in input_data_x:
        print(f"[{input[0]}, {input[1]}]")
    print("Expected output:")
    for expected_output_res in expected_input_results:
        print(f"[{expected_output_res[0]}] ")
    print("Data validation input:")
    for val_input in data_validation_input:
        print(f"[{val_input[0]}, {val_input[1]}]")
    print("Expected data validation results")
    for exp_val_res in expected_data_validation_results:
        print(f"[{exp_val_res[0]}]")

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
    print_inputs(input_data_x, expected_input_results, data_validation_input, expected_data_validation_results);
    # k = 4
    # xor_neural_network(train_data, expected_train_res, val_data, exp_val_res, k, bridge):
    bridge_options = [True, False]
    k_options = [2, 4]
    learning_rate_options = [0.1, 0.01]
    text_file = open("output.txt", "w")
    exp_num = 0
    # def run_experiment(text_file, exp_num, train_data, expected_train_res, val_data, exp_val_res, k, bridge, learning_rate):
    for bridge in bridge_options:
        for k in k_options:
            for learning_rate in learning_rate_options:
                run_experiment(text_file, exp_num, input_data_x, expected_input_results, data_validation_input,
                               expected_data_validation_results, k, bridge, learning_rate)
