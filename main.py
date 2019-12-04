#EX2 XOR using Gradient Descent and Cross Entropy loss
#1 1 --> 0
#1 0 --> 1
#0 1 --> 1
#0 0 --> 0
#Daniil Rolnik
#334018009
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

max_num_of_steps = 40000
num_max_succ_runs = 10
temperature = 0.4
min_val_loss = 0.2
huge_num = 1000000
min_change = 0.0001

def get_output(x, w1, w2, bridge, b1, b2):
    ##############################################################################
    ### Get target output  ###
    #print(f"x shape = {tf.shape(x)}, w1 shape = {tf.shape(w1)}")
    z1 = tf.add(tf.matmul(x, w1),b1)
    #print("after got z1")
    temp = tf.sigmoid(z1 / temperature)
    if bridge:
        hidden_layer_res = tf.concat([temp, x], 1)
    else:
        hidden_layer_res = temp
    #print(f"hidden layer res shape = {tf.shape(hidden_layer_res)}, w2 shape = {tf.shape(w2)}")
    z2 = tf.add(tf.matmul(hidden_layer_res, w2), b2)
    #print("after got z2")

    output = tf.sigmoid(z2 / temperature)
    return output


def perform_steps(sess, train_grad, x, y, train_data, expected_train_res, loss, val_data, exp_val_res):
    ##############################################################################
    ### Perform epoch steps  ###
    success, num_of_succ, last_loss = (False, 0, huge_num)
    for step in range(max_num_of_steps):
        # print(f"step = {step}")
        sess.run(train_grad, {x: train_data, y: expected_train_res})
        val_loss = sess.run(loss, {x: val_data, y: exp_val_res})
        # print(f"val_loss = {val_loss}")
        # print(f"last_loss = {last_loss}")
        if abs(last_loss - val_loss) < min_change:
            num_of_succ = num_of_succ + 1
            # print(f"num_of_succ = {num_of_succ}")
            if val_loss < min_val_loss and num_of_succ >= num_max_succ_runs:
                success = True
                break
        else:
            num_of_succ = 0
        last_loss = val_loss
    return success, step, val_loss


def xor_neural_network(train_data, expected_train_res, val_data, exp_val_res, k, bridge, learning_rate):
    ##############################################################################
    ### Perform xor neural network  ###

    amount_input_neurons, amount_output_neurons, rand_seed = (len(train_data[0]), 1, 350)
    # print(f"amount_input_neurons = {amount_input_neurons}, amount_output_neurons = {amount_output_neurons}")
    x = tf.compat.v1.placeholder(tf.float32, [None, amount_input_neurons])
    y = tf.compat.v1.placeholder(tf.float32, [None, amount_output_neurons])
    w1 = tf.Variable(tf.random.uniform([amount_input_neurons, k], minval=-1, maxval=1, seed=0),
                                       dtype=tf.dtypes.float32, name=None)
    if bridge == True:
        w2_num_of_rows = k + amount_input_neurons
    else:
        w2_num_of_rows = k
    w2 = tf.Variable(tf.random.uniform([w2_num_of_rows, 1], minval=-1, maxval=1, seed=rand_seed),dtype=tf.dtypes.float32,  name=None)
    b1 = tf.compat.v1.Variable(tf.random.uniform([1, k], minval=-1, maxval=1, seed=rand_seed), dtype=tf.dtypes.float32, name=None)
    b2 = tf.compat.v1.Variable(tf.random.uniform([1, 1], minval=-1, maxval=1, seed=rand_seed), dtype=tf.dtypes.float32, name=None)
    sess = tf.compat.v1.Session()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    output = get_output(x, w1, w2, bridge, b1, b2)
    loss = - tf.reduce_sum((y * tf.math.log(output)) + (1 - y) * tf.math.log(1.0 - output))
    train_grad = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    success, step, val_loss = \
        perform_steps(sess, train_grad, x, y, train_data, expected_train_res, loss, val_data, exp_val_res);
    new_b1, new_b2, new_w1, new_w2, new_loss = \
        sess.run([b1, b2, w1, w2, loss], {x: val_data, y: expected_data_validation_results})
    train_loss = sess.run(loss, {x: train_data, y: expected_train_res})
    train = sess.run(loss, {x: train_data, y: expected_train_res})
    return new_b1, new_b2, new_w1, new_w2, new_loss, val_loss, train_loss, success, step


def run_experiment(text_file, exp_num, train_data, expected_train_res, val_data, exp_val_res, k, bridge, learning_rate):
    steps, succ_count, fail_count, all_val_losses, all_train_losses = ([], 0, 0, [], [])
    ##############################################################################
    ### Run all predefined experiments  ###
    while succ_count < num_max_succ_runs:
        b1, b2,  w1, w2, loss, val_loss,  train_loss, success, step = xor_neural_network(train_data, expected_train_res, val_data, exp_val_res, k, bridge, learning_rate)
        if success == True:
            steps.append(step)
            all_val_losses.append(val_loss)
            all_train_losses.append(train_loss)
            succ_count = succ_count + 1
            print(f"succ_count = {succ_count}")
        else:
            fail_count = fail_count + 1
            print(f"fail_count = {fail_count}")
    mean_epochs = np.mean(steps)
    std_epochs = np.std(steps)
    mean_val_loss = np.mean(all_train_losses)
    std_val_loss_percent = np.std(all_val_losses)
    mean_train_loss = np.mean(all_train_losses)
    std_train_loss_percent = np.std(train_loss)
    write_experiment(text_file, exp_num, k, learning_rate, bridge, mean_epochs, std_epochs, fail_count, mean_val_loss,
                     std_val_loss_percent, mean_train_loss, std_train_loss_percent)


def write_experiment(text_file, exp_num, k, learning_rate, bridge, mean_epochs, std_epochs, fail_count,
                     mean_val_loss, std_val_loss_percent, mean_train_loss, std_train_loss_percent):
    ##############################################################################
    ### Write experiment result to the console and to the text file  ###
    print("writing")
    result_str = f'experiment{exp_num}:hidden:{k}, LR:{learning_rate}, bridge:{bridge}\n' \
                 f'mean_epocs:{mean_epochs}, std/epocsPerc {std_epochs},Failures = {fail_count}\n' \
                 f'mean_valid_loss:{mean_val_loss}, stdvalidlossPercent: {std_val_loss_percent},\n' \
                 f'meanTrainLoss: {mean_train_loss}, stdTrainLossPercent:{std_train_loss_percent}\n\n'
    print(result_str)
    try:
        text_file.write(result_str)
        text_file.flush()
    except IOError:
        print("Problem with writing to file")


def print_inputs(input_data_x, expected_input_results, data_validation_input, expected_data_validation_results):
    ##############################################################################
    ### Print all given inputs  ###
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
    ### Training Examples
    ### All combinations of XOR
    input_data_x = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])
    # expected results from activating XOR (corresponds to the lists in input_train list)
    expected_input_results = np.array([[0], [1], [1], [0]])
    data_validation_input = np.array([[1, 0.1],
                             [1, 0.9],
                             [0.9, 0.9],
                             [0.1, 0.9]])
    expected_data_validation_results = np.array([[1], [0], [0], [1]])
    print_inputs(input_data_x, expected_input_results, data_validation_input, expected_data_validation_results);
    bridge_options = [True, False]
    k_options = [2, 4]
    learning_rate_options = [0.1, 0.01]
    text_file = open("output.txt", "w")
    exp_num = 1
    for bridge in bridge_options:
        for k in k_options:
            for learning_rate in learning_rate_options:
                run_experiment(text_file, exp_num, input_data_x, expected_input_results, data_validation_input,
                               expected_data_validation_results, k, bridge, learning_rate)
                exp_num = exp_num + 1

    text_file.close()
    print("finished program and closed output file")
