import math
import os.path
import random
import sys
import time
import csv
import ReadHealper as read
import numpy as np
import random

# hyperparameter
learning_rate = 0.03
momentum_rate = 0.1
mini_bath_size = 1
epoch = 200
relu_max = 2

# activation and derivative functions
def sigmoid(x):
    return 1.0 / (math.exp((-x)) + 1.0)

def sigmoid_d(x):
    return (1.0 - x) * x

def tanh(x):
    return math.tanh(x)

def tanh_d(x):
    t = tanh(x)
    return 1 - pow(t, 2)

def softmax(x):
    t = np.exp(x)
    return t / np.sum(t)

def relu(x):
    x = x * (x <= relu_max) + (x > relu_max) * relu_max
    return x * (x >=0)

def relu_d(x):
    return x > 0 +0.01

# Loss function
# binary cross entropy
def cross_entropy(t, p):
    return -(t * math.log(p, 10) + (1 - t) * math.log(1-p, 10))

def cross_entropy_d(t, p):
    return t / p + (1 - t)/(1 - p)

def MSE(t, p):
    return (t - p)**2 * 0.5

def MSE_d(t, p):
    return (t - p)


# create a x by y matrix
def build_matrix(x, y):
    re = []
    for i in range(x):
        re.append([0] * y)
    return re


# initialize matrix with random num between x and y
def rand_matrix(matrix, x, y):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix[i][j] = (y - x) * random.random() + x


def rand_array(arr, x, y):
    for i in range(len(arr)):
        arr[i] = (y - x) * random.random() + x


class NN:
    def __init__(self, feature_input, label_input, test_input):
        # init case number groups
        self.case_list = []
        for i in range(len(feature_input)):
            self.case_list.append(i)

        # store case, label and test
        self.features = feature_input
        self.labels = label_input
        self.tests = test_input

        # input, hidden and output layer nodes number
        self.input_nodes = len(feature_input[0]) + 1
        self.hidden1_nodes = 6
        self.hidden2_nodes = 2
        self.output_nodes = 1

        # input, hiddens and output layer node matrix
        self.input = self.input_nodes * [1.0]
        self.hidden1 = self.hidden1_nodes * [1.0]
        self.hidden2 = self.hidden2_nodes * [1.0]
        self.output = self.output_nodes * [1.0]

        # # create correction matrix
        self.input_M = build_matrix(self.input_nodes, self.hidden1_nodes)
        self.hidden1_M = build_matrix(self.hidden1_nodes, self.hidden2_nodes)
        self.hidden2_M = build_matrix(self.hidden2_nodes, self.output_nodes)

        # weights of each layers and initialize weight matrix
        self.input_weight = build_matrix(self.input_nodes, self.hidden1_nodes)
        rand_matrix(self.input_weight, -0.4, 0.4)
        self.hidden1_weight = build_matrix(self.hidden1_nodes, self.hidden2_nodes)
        rand_matrix(self.hidden1_weight, -0.4, 0.4)
        self.hidden2_weight = build_matrix(self.hidden2_nodes, self.output_nodes)
        rand_matrix(self.hidden2_weight, -0.4, 0.4)

        # bias of neural for each layer
        # self.layer1_bias = self.hidden1_n * [1]
        # rand_array(self.layer1_bias, -0.5, 0.5)
        # self.layer2_bias = self.hidden2_n * [1]
        # rand_array(self.layer2_bias, -0.5, 0.5)
        # self.output_bias = self.output_n * [1]
        # rand_array(self.output_bias, -0.5, 0.5)

    # forward direction to predict
    def predict(self, features):
        # add features to input matrix
        for i in range(self.input_nodes - 1):
            self.input[i] = features[i]
        # the last node is for bias, initialize it to 1
        self.input[self.input_nodes - 1] = 1

        # calculate hidden layer1
        for j in range(self.hidden1_nodes):
            sum = 0
            for i in range(self.input_nodes):
                sum += self.input_weight[i][j] * self.input[i]
            # self.hidden1[j] = tanh(sum)
            self.hidden1[j] = sigmoid(sum)

        # calculate hidden layer2
        for j in range(self.hidden2_nodes):
            sum = 0
            for i in range(self.hidden1_nodes):
                sum += self.hidden1_weight[i][j] * self.hidden1[i]
            # self.hidden2[j] = tanh(sum)
            self.hidden2[j] = sigmoid(sum)

        # calculate output
        for j in range(self.output_nodes):
            sum = 0
            for i in range(self.hidden2_nodes):
                sum += self.hidden2_weight[i][j] * self.hidden2[i]
            # self.output[j] = tanh(sum)
            self.output[j] = sigmoid(sum)

        # return prediction
        return self.output[0]

    # back propagation
    def back_propagate(self, label):
        # output error calculation
        # output_del = [tanh_d(self.output[0]) * avg_err]
        # output_del = self.output_nodes * [0.0]
        # for i in range(self.output_nodes):
        #     # err = label[i] - self.output[i]
        #     err = MSE_d(label[i], self.output[i])
        #     # output_del[i] = tanh_d(self.output[i]) * err
        #     output_del[i] = sigmoid_d(self.output[i]) * err

        output_del = [sigmoid_d(self.output[0]) * MSE_d(label[0], self.output[0])]

        # hidden layer2 error calculation
        hidden2_del = self.hidden2_nodes * [0.0]
        for i in range(self.hidden2_nodes):
            err = 0.0
            for j in range(self.output_nodes):
                err += self.hidden2_weight[i][j] * output_del[j]
            # hidden2_del[i] = tanh_d(self.hidden2[i]) * err
            hidden2_del[i] = sigmoid_d(self.hidden2[i]) * err

        # hidden layer1 error calculation
        hidden1_del = [0.0] * self.hidden1_nodes
        for i in range(self.hidden1_nodes):
            err = 0.0
            for j in range(self.hidden2_nodes):
                err += self.hidden1_weight[i][j] * hidden2_del[j]
            # hidden1_del[i] = tanh_d(self.hidden1[i]) * err
            hidden1_del[i] = sigmoid_d(self.hidden1[i]) * err

        # update hidden2_layer weights
        for i in range(self.hidden2_nodes):
            for j in range(self.output_nodes):
                change = output_del[j] * self.hidden2[i]
                self.hidden2_weight[i][j] += momentum_rate * self.hidden2_M[i][j] + learning_rate * change
                self.hidden2_M[i][j] = change

        # update hidden1_layer weights
        for i in range(self.hidden1_nodes):
            for j in range(self.hidden2_nodes):
                change = hidden2_del[j] * self.hidden1[i]
                self.hidden1_weight[i][j] += momentum_rate * self.hidden1_M[i][j] + learning_rate * change
                self.hidden1_M[i][j] = change

        # update input weights
        for i in range(self.input_nodes):
            for j in range(self.hidden1_nodes):
                change = hidden1_del[j] * self.input[i]
                self.input_weight[i][j] += momentum_rate * self.input_M[i][j] + learning_rate * change
                self.input_M[i][j] = change

        # get total error
        # err = 0.0
        # for i in range(len(label)):
            # err += 0.5 * (label[i] - self.output[i]) ** 2
            # err += MSE(label[i], self.output[i])
            # err += cross_entropy(label[i], self.output[i])
        err = MSE(label[0], self.output[0])
        print(err)
        return err


    # training
    def train(self):
        # # mini-batch training
        # for i in range(epoch):
        #     random.shuffle(self.case_list)
        #     ptr = 0
        #     sum_err = 0.0
        #     while ptr < len(self.case):
        #         temp = 0
        #         err = 0.0
        #         count = 0
        #         # get average err of the batch
        #         while (ptr < len(self.case) and temp < mini_bath_size):
        #             case_no = self.case_list[ptr]
        #             c = self.case[case_no]
        #             l = self.label[case_no]
        #             self.predict(c)
        #             err += l - self.output[0]
        #             temp += 1
        #             ptr += 1
        #             count += 1
        #         err /= count
        #         sum_err += err
        #         #  update weights
        #         self.back_propagate(err)
        #         print(f"Total error is {sum_err}")

        for j in range(epoch):
            err = 0.0
            for i in range(len(self.features)):
                label = [self.labels[i]]
                case = self.features[i]
                self.predict(case)
                err += self.back_propagate(label)
            print(err)

    def make_prediction(self):
        # numbe of errors compared with true labels
        no_err = 0
        # train model with input training data and label
        self.train()


        # read in test labels for prediction comparision
        # test_label_path = "./xor_test_label.csv"
        # test_label = read.readTestLabel(test_label_path)

        # make predictions and write to output file
        # print("Below is prediction")
        # for i in range(len(test_label)):
        #     c = self.tests[i]
        #     pred = self.predict(c)
        #     if abs(pred - 0) < abs(1 - pred):
        #         if 0 != test_label[i]:
        #             no_err += 1
        #     else:
        #         if 1 != test_label[i]:
        #
        #             no_err += 1

        # check if output file exist, remove it if it does
        if os.path.exists('test_predictions.csv'):
            os.remove('')
        with open('test_predictions.csv', 'a') as f:
            for t in self.tests:
                pred = self.predict(t)
                if abs(pred - 0) < abs (1 - pred):
                    pred_l  = '0\n'
                else:
                    pred_l = '1\n'
                line = [pred_l]
                f.writelines(line)

        # print(f"No of erroes is {no_err}")
        # print(f"Correct rate is {1 - no_err/len(test_label)}")


if __name__ == '__main__':
    ts = time.time()
    case, label, test = read.readInput(sys.argv[1:])
    nn = NN(case, label, test)
    nn.train()
    nn.make_prediction()
    te = time.time()
    print(f"Total time used is {te-ts}")