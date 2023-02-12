import math
import random

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class BPNeuralNetwork:
    def __init__(self):
        self.input_n = 0
        self.hidden1_n = 0
        self.hidden2_n = 0
        self.output_n = 0

        self.input_cells = []
        self.hidden1_cells = []
        self.hidden2_cells = []
        self.output_cells = []

        self.output_weights = []
        self.hidden1_weights = []
        self.hidden2_weights = []

        self.output_correction = []
        self.layer2_correction = []
        self.layer1_correction = []

    def setup(self, ni, nh1, nh2, no):
        self.input_n = ni + 1
        self.hidden1_n = nh1
        self.hidden2_n = nh2
        self.output_n = no

        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden1_cells = [1.0] * self.hidden1_n
        self.hidden2_cells = [1.0] * self.hidden2_n
        self.output_cells = [1.0] * self.output_n

        # init weights
        self.hidden1_weights = make_matrix(self.input_n, self.hidden1_n)
        self.hidden2_weights = make_matrix(self.hidden1_n, self.hidden2_n)
        self.output_weights = make_matrix(self.hidden2_n, self.output_n)

        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden1_n):
                self.hidden1_weights[i][h] = rand(-0.2, 0.2)

        for h in range(self.hidden1_n):
            for o in range(self.hidden2_n):
                self.hidden2_weights[h][o] = rand(-2.0, 2.0)

        for h in range(self.hidden2_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

        # init correction matrix
        self.hidden1_correction = make_matrix(self.input_n, self.hidden1_n)
        self.hidden2_correction = make_matrix(self.hidden1_n, self.hidden2_n)
        self.output_correction = make_matrix(self.hidden2_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]

        # activate hidden layer1
        for j in range(self.hidden1_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.hidden1_weights[i][j]
            self.hidden1_cells[j] = sigmoid(total)

        # activate hidden layer2
        for j in range(self.hidden2_n):
            total = 0.0
            for i in range(self.hidden1_n):
                total += self.hidden1_cells[i] * self.hidden2_weights[i][j]
            self.hidden2_cells[j] = sigmoid(total)

        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden2_n):
                total += self.hidden2_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)
        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)
        # get output layer error
        output_deltas = [0.0] * self.output_n
        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error

        # get hidden layer2 error
        hidden2_deltas = [0.0] * self.hidden2_n
        for h in range(self.hidden2_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden2_deltas[h] = sigmoid_derivative(self.hidden2_cells[h]) * error


        # get hidden layer1 error
        hidden1_deltas = [0.0] * self.hidden1_n
        for h in range(self.hidden1_n):
            error = 0.0
            for o in range(self.hidden2_n):
                error += hidden2_deltas[o] * self.hidden2_weights[h][o]
            hidden1_deltas[h] = sigmoid_derivative(self.hidden1_cells[h]) * error

        # update output weights
        for h in range(self.hidden2_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden2_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        # update input weights
        for i in range(self.hidden1_n):
            for h in range(self.hidden2_n):
                change = hidden2_deltas[h] * self.hidden1_cells[i]
                self.hidden2_weights[i][h] += learn * change + correct * self.hidden2_correction[i][h]
                self.hidden2_correction[i][h] = change

        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden1_n):
                change = hidden1_deltas[h] * self.input_cells[i]
                self.hidden1_weights[i][h] += learn * change + correct * self.hidden1_correction[i][h]
                self.hidden1_correction[i][h] = change


        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def train(self, cases, labels, limit=10000, learn=0.05, correct=0.1):
        for j in range(limit):
            # print(f'epoch {j}')
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)

    def test(self):
        cases = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ]
        labels = [[0], [1], [1], [0]]
        self.setup(2, 7, 5, 1)
        self.train(cases, labels, 15000, 0.05, 0.1)
        for case in cases:
            print(self.predict(case))


if __name__ == '__main__':
    nn = BPNeuralNetwork()
    nn.test()
