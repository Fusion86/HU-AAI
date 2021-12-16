import copy
import random
import numpy as np
from typing import List


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def binary_step(x):
    return 1.0 if x >= 0 else 0.0


def random_weights(dim):
    return np.random.uniform(-1, 1, dim)


class NeuralNetworkNode:
    def __init__(self, inputs=None):
        self.inputs = inputs

    @property
    def output(self):
        raise NotImplementedError


class InputNode(NeuralNetworkNode):
    @property
    def output(self):
        return self.inputs


class Perceptron(NeuralNetworkNode):
    def __init__(
        self, inputs=None, weights=None, bias=None, activation_func=binary_step
    ):
        super().__init__(inputs)
        self.weights = weights
        self.bias = bias
        self.activation_func = activation_func
        self.z = None
        self.delta = None

        if self.weights == None and self.bias == None:
            self.randomize()

    @property
    def output(self):
        return self.a

    @property
    def a(self):
        return self.activation_func(self.z)

    def activation_func_derivative(self, x):
        if self.activation_func == sigmoid:
            return sigmoid(x) * (1 - sigmoid(x))
        else:
            raise NotImplementedError

    def randomize(self):
        self.bias = random.uniform(-1, 1)
        self.weights = random_weights(len(self.inputs))

    def update(self):
        if len(self.inputs) != len(self.weights):
            raise ArithmeticError("len(inputs) != len(weights)")
        inputs = [x.output for x in self.inputs]
        self.z = np.dot(inputs, self.weights) + self.bias


class NeuralNetworkLayer:
    def __init__(self):
        self.nodes: List[NeuralNetworkNode] = []
        # self.deltas = []

    # def reset_deltas(self):
    #     self.deltas = []
    #     for _ in range(len(self.nodes)):
    #         self.deltas.append(None)

    @property
    def output(self):
        return [x.output for x in self.nodes]

    def __len__(self):
        return len(self.nodes)

    def __iter__(self):
        return iter(self.nodes)


class InputLayer(NeuralNetworkLayer):
    def __init__(self, neuron_count):
        super().__init__()
        self.nodes = [InputNode() for _ in range(neuron_count)]


class PerceptronLayer(NeuralNetworkLayer):
    def __init__(self, neuron_count, inputs, activation_func):
        super().__init__()
        self.nodes = [
            Perceptron(inputs, activation_func=activation_func)
            for _ in range(neuron_count)
        ]


class NeuralNetwork:
    def __init__(self, layer_spec, activation_func=sigmoid):
        self.layers: List[NeuralNetworkLayer] = []

        # Input layer
        self.layers.append(InputLayer(layer_spec[0]))

        # Hidden layers
        for i, neuron_count in enumerate(layer_spec[1]):
            self.layers.append(
                PerceptronLayer(neuron_count, self.layers[i], activation_func)
            )

        # Output layer
        self.layers.append(
            PerceptronLayer(layer_spec[2], self.layers[-1], activation_func)
        )

    @property
    def network_structure_str(self):
        x = "I.{}".format(len(self.layers[0]))

        for layer in range(1, len(self.layers) - 1):
            x += " - H.{}".format(len(self.layers[layer]))

        x += " - O.{}".format(len(self.layers[-1]))
        return x

    @property
    def output(self):
        return self.layers[-1].output

    def execute(self, inputs: list):
        self.load_inputs(inputs)
        self.feedforward()
        return self.output

    def load_inputs(self, inputs: list):
        if len(inputs) != len(self.layers[0]):
            raise Exception("Input count does not match the number of input nodes.")

        for i, v in enumerate(inputs):
            self.layers[0].nodes[i].inputs = v

    def feedforward(self):
        for _, layer in enumerate(self.layers[1:]):
            for node in layer.nodes:
                node.update()

    def backpropagation_last_layer(self, desired_output):
        last_layer = self.layers[-1]
        for i, node in enumerate(last_layer):
            diff = desired_output[i] - node.a
            node.delta = node.activation_func_derivative(node.z) * diff

    def backpropagation_other_layer(self, layer_index):
        for i, node in enumerate(self.layers[layer_index]):
            total = 0
            for j, node_j in enumerate(self.layers[layer_index + 1]):
                total += node_j.delta * node_j.weights[i]
            node.delta = node.activation_func_derivative(node.z) * total

    def update_weights_and_biases(self, learning_factor):
        for layer_idx in range(2, len(self.layers)):
            for j, node in enumerate(self.layers[layer_idx]):
                node.bias += learning_factor * node.delta

                for i, node_input in enumerate(node.inputs):
                    node.weights[i] += learning_factor * node.delta * node_input.a

    def train(self, training_set, labels, iterations, learning_factor):
        for _ in range(iterations):
            for training_row in training_set:
                # 0. Load input values.
                self.load_inputs(training_row[0])

                # 1. Feed forward.
                self.feedforward()

                # 2. Backpropagation of the last layer.
                desired_output = [0 for _ in range(len(labels))]
                desired_output[labels.index(training_row[1])] = 1

                self.backpropagation_last_layer(desired_output)

                # 3. Backpropagation of the other layers, from last to first.
                # TODO: Implement range loop instead of hardcoded
                for layer_index in reversed(range(1, len(self.layers) - 1)):
                    self.backpropagation_other_layer(layer_index)

                # 4. Update weights and biases
                self.update_weights_and_biases(learning_factor)

    def validate(self, validation_set, labels):
        score = 0

        for row in validation_set:
            output = self.execute(row[0])
            label = output.index(max(output))
            expected_label = labels.index(row[1])

            if label == expected_label:
                score += 1

        return score / len(validation_set)