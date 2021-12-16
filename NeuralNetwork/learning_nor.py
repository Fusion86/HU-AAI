import random
import numpy as np
import matplotlib.pyplot as plt
from neuron import Perceptron, InputNode, sigmoid, gen_weights

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    inputs = [
        InputNode(0),
        InputNode(0),
    ]

    desired_output = 1
    learning_factor = 1

    nor_node = Perceptron(inputs, gen_weights(1), 1, sigmoid)

    iterations = []
    outputs = []

    # for iteration in range(2000):
    #     nor_node.delta_update(desired_output)

    plt.title("Learning Easy (learning factor = {})".format(learning_factor))
    plt.plot(iterations, outputs)
    plt.xlabel("Iteration")
    plt.ylabel("Output")
    plt.savefig("learning_easy.png")
