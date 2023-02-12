import math
import random
import sys
import time

import ReadHealper as read
import numpy as np
import random

# hyperparameter
learning_rate = 0.03
correctio_rate = 0.1
mini_bath_size = 1
epoch = 300
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

# def relu(x):
#     x = x * (x <= relu_max) + (x > relu_max) * relu_max
#     return x * (x >=0)
#
# def relu_d(x):
#     return x > 0 +0.01

# Loss function
# binary cross entropy
def cross_entropy(t, p):
    return -(t * math.log(p, 10) + (1 - t) * math.log(1-p, 10))

def cross_entropy_d(t, p):
    return t / p + (1 - t)/(1 - p)


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
    def __init__(self, case, label, test):
        # init case number groups
        self.case_list = []
        for i in range(len(case)):
            self.case_list.append(i)

        # store case, label and test
        self.case = case
        self.label = label
        self.test = test

        # input, hidden and output layer nodes number
        self.input_n = len(case[0]) + 1
        self.hidden1_n = 7
        self.hidden2_n = 3
        self.output_n = 1

        # input, hiddens and output layer node matrix
        self.input = [1.0] * self.input_n
        self.hidden1 = [1.0] * self.hidden1_n
        self.hidden2 = [1.0] * self.hidden2_n
        self.output = [1.0] * self.output_n

        # # create correction matrix
        self.input_correction = build_matrix(self.input_n, self.hidden1_n)
        self.hidden1_correction = build_matrix(self.hidden1_n, self.hidden2_n)
        self.hidden2_correction = build_matrix(self.hidden2_n, self.output_n)

        # weights of each layers and initialize weight matrix
        self.input_weight = build_matrix(self.input_n, self.hidden1_n)
        rand_matrix(self.input_weight, -0.2, 0.2)
        self.hidden1_weight = build_matrix(self.hidden1_n, self.hidden2_n)
        rand_matrix(self.hidden1_weight, -0.2, 0.2)
        self.hidden2_weight = build_matrix(self.hidden2_n, self.output_n)
        rand_matrix(self.hidden2_weight, -0.2, 0.2)

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
        for i in range(self.input_n - 1):
            self.input[i] = features[i]
        # the last node is for bias, initialize it to 0.5
        self.input[self.input_n - 1] = 0.5

        # calculte hidden layer1
        for j in range(self.hidden1_n):
            sum = 0
            for i in range(self.input_n):
                sum += self.input[i] * self.input_weight[i][j]
            self.hidden1[j] = tanh(sum)
            # self.hidden1[j] = sigmoid(sum)

        # calculate hidden layer2
        for j in range(self.hidden2_n):
            sum = 0
            for i in range(self.hidden1_n):
                sum += self.hidden1[i] * self.hidden1_weight[i][j]
            self.hidden2[j] = tanh(sum)
            # self.hidden2[j] = sigmoid(sum)

        # calculate output
        for j in range(self.output_n):
            sum = 0
            for i in range(self.hidden2_n):
                sum += self.hidden2[i] * self.hidden2_weight[i][j]
            self.output[j] = tanh(sum)
            # self.output[j] = sigmoid(sum)

        # return prediction
        return self.output[0]

    # back propagation
    def back_propagate(self, output_del):
        # output error calculation
        # output_del = [tanh_d(self.output[0]) * avg_err]
        # output_del = [0.0] * self.output_n
        # for i in range(self.output_n):
        #     err = label[i] - self.output[i]
            # output_del[i] = tanh_d(self.output[i]) * err
            # output_del[i] = sigmoid_d(self.output[i]) * err

        # hidden layer2 error calculation

        hidden2_del = [0.0] * self.hidden2_n
        for i in range(self.hidden2_n):
            err = 0.0
            for j in range(self.output_n):
                err += output_del[j] * self.hidden2_weight[i][j]
            hidden2_del[i] = tanh_d(self.hidden2[i]) * err
            # hidden2_del[i] = sigmoid(self.hidden2[i]) * err

        # hidden layer1 error calculation
        hidden1_del = [0.0] * self.hidden1_n
        for i in range(self.hidden1_n):
            err = 0.0
            for j in range(self.hidden2_n):
                err += hidden2_del[j] * self.hidden1_weight[i][j]
            hidden1_del[i] = tanh_d(self.hidden1[i]) * err
            # hidden1_del[i] = sigmoid_d(self.hidden1[i]) * err

        # update hidden2_layer weights
        for i in range(self.hidden2_n):
            for j in range(self.output_n):
                change = output_del[j] * self.hidden2[i]
                self.hidden2_weight[i][j] += learning_rate * change + correctio_rate * self.hidden2_correction[i][j]
                self.hidden2_correction[i][j] = change

        # update hidden1_layer weights
        for i in range(self.hidden1_n):
            for j in range(self.hidden2_n):
                change = hidden2_del[j] * self.hidden1[i]
                self.hidden1_weight[i][j] += learning_rate * change + correctio_rate * self.hidden1_correction[i][j]
                self.hidden1_correction[i][j] = change

        # update input weights
        for i in range(self.input_n):
            for j in range(self.hidden1_n):
                change = hidden1_del[j] * self.input[i]
                self.input_weight[i][j] += learning_rate * change + correctio_rate * self.input_correction[i][j]
                self.input_correction[i][j] = change

        # get total error
        # err = 0.0
        # for i in range(len(label)):
        #     err += 0.5 * (label[i] - self.output[i]) ** 2
        #     # err += cross_entropy(label[i], self.output[i])
        # print(err)
        # return err


    # training
    def train(self):
        # # mini-batch training
        for i in range(epoch):
            random.shuffle(self.case_list)
            ptr = 0
            sum_err = 0.0
            while ptr < len(self.case):
                temp = 0
                err = 0.0
                count = 0
                output_grad_temp = []

                # get average err of the batch
                while (ptr < len(self.case) and temp < mini_bath_size):
                    case_no = self.case_list[ptr]
                    c = self.case[case_no]
                    l = self.label[case_no]
                    self.predict(c)
                    err = l - self.output[0]
                    output_grad_temp.append(tanh_d(self.output[0]) * err)
                    temp += 1
                    ptr += 1
                    count += 1
                # err /= count
                # sum_err += err
                #  update weights
                output_grad_final = [np.mean(np.array(output_grad_temp))]
                self.back_propagate(output_grad_final)
                # print(f"Total error is {sum_err}")
                print(output_grad_final)


        # for j in range(epoch):
        #     err = 0
        #     for i in range(len(self.case)):
        #         label = [self.label[i]]
        #         case = self.case[i]
        #         self.predict(case)
        #         err += self.back_propagate(label)
        #     print(err)

    def make_prediction(self):
        no_err = 0
        self.train()
        test_label_path = "./spiral_train_label.csv"
        test_label = read.readTestLabel(test_label_path)
        print("Below is prediction")
        for i in range(len(self.case)):
            c = self.case[i]
            pred = self.predict(c)
            if abs(pred - 0) < abs(1 - pred):
                if 0 != test_label[i]:
                    no_err += 1
            else:
                if 1 != test_label[i]:
                    no_err += 1


        print(f"No of erroes is {no_err}")
        print(f"Error ercentage is {no_err/len(test_label)}")


if __name__ == '__main__':
    ts = time.time()
    case, label, test = read.readInput(sys.argv[1:])
    # case = [
    #     [0, 0],
    #     [0, 1],
    #     [1, 0],
    #     [1, 1],
    # ]
    # label = [[0], [1], [1], [0]]
    # test = [
    #     [0, 0],
    #     [0, 1],
    #     [1, 0],
    #     [1, 1],
    # ]
    nn = NN(case, label, test)
    nn.train()
    nn.make_prediction()
    te = time.time()
    print(f"Total time used is {te-ts}")