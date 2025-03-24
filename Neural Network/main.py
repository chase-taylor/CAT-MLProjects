import numpy as np
import pandas as pd
import sys
import time


class Node:
    # initializes all node attributes
    def __init__(self, num_inputs):
        self.weights = [0 for i in range(num_inputs + 1)]
        self.inputs = None
        self.output = None
        self.output_table = None
        self.error = None

    # determines what the node should output for a given input
    def calculate_output(self, input_arr):
        output = 0
        # insert bias into inputs
        self.inputs = [1]
        self.inputs[1:] = input_arr
        for i in range(len(self.inputs)):
            output += self.inputs[i] * self.weights[i]
        self.output = 1/(1+np.e**(-output))
        return self.output

    # used for forward pass of table
    def calculate_output_table(self, df):
        temp = df.copy()
        temp.insert(loc=0, column='bias',value=[1 for i in range(temp.shape[0])])
        for i in range(temp.shape[1]):
            temp.iloc[:,i] *= self.weights[i]
        temp = temp.sum(axis=1)
        for i in range(temp.shape[0]):
            temp.iat[i] = 1 / (1 + np.e ** (-(temp.iat[i])))
        self.output_table = temp
        return self.output_table

    # updates all weights in the node
    def update_weights(self, lr):
        for i in range(len(self.weights)):
            self.weights[i] += lr * self.error * self.inputs[i]

# initializes all nodes in the neural network
def init_network(num_inputs, num_hidden_layers, num_hidden_nodes):
    network = list()
    for i in range(num_hidden_layers):
        layer = [Node(num_inputs) for j in range(num_hidden_nodes)]
        network.append(layer)
        num_inputs = num_hidden_nodes
    network.append([Node(num_inputs)])
    return network

# determines the output from the network for a given input
def forward_pass(network, row):
    # sets the output from each given layer in the network to the input for the next layer until reaches end of network
    for layer in network:
        output = []
        for node in layer:
            output.append(node.calculate_output(row))
        row = output
    return output[0]

# performs a forward pass using an entire table then calculates mean square error to reduce number of function calls
def forward_pass_table(network, df, num_hidden_layers, num_hidden_nodes):
    temp = df.iloc[:,:-1].copy()
    mt = pd.DataFrame({})
    for layer in network:
        layer_table = mt.copy()
        node_num = 0
        for node in layer:
            layer_table.insert(node_num, node_num, node.calculate_output_table(temp))
            node_num += 1
        temp = layer_table
    val = temp.shape[0]
    temp.iloc[:,0] -= df.iloc[:,-1]
    temp.iloc[:,0] *= temp.iloc[:,0]
    return temp.sum()/val

# calculates the derivative of the sigmoid function given its result
def d_output(output):
    return output * (1 - output)

# backpropagation algorithm
def backward_pass(network, output, expected, lr):
    # setting output node error
    network[len(network)-1][0].error = d_output(output) * (expected - output)
    # setting error for each hidden node
    for layer in range(len(network)-2, -1, -1):
        for node_i_index in range(len(network[layer])):
            total = 0
            for node_j in network[layer+1]:
                total += node_j.weights[node_i_index+1] * node_j.error
            network[layer][node_i_index].error = d_output(network[layer][node_i_index].output) * total
    # updating weights
    for layer in network:
        for node in layer:
            node.update_weights(lr)

# runs the program with the provided attributes
def run(train, test, num_hidden_layers, num_hidden_nodes, learn_rate, num_iter):
    network = init_network(train.shape[1] - 1, num_hidden_layers, num_hidden_nodes)
    for i in range(num_iter):
        print("At iteration " + str(i+1) + ":")
        row_index = i % train.shape[0]
        output = forward_pass(network, train.iloc[row_index,:-1])
        print("Forward pass output: %.4f" % output)
        backward_pass(network, output, train.iat[row_index, -1], learn_rate)
        print("Average squared error on training set (" + str(train.shape[0]) + " instances): %.4f" % forward_pass_table(network, train, num_hidden_layers, num_hidden_nodes))
        print("Average squared error on test set (" + str(test.shape[0]) + " instances): %.4f" % forward_pass_table(network, test, num_hidden_layers, num_hidden_nodes))
        print()


run(pd.read_table(sys.argv[1]), pd.read_table(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))