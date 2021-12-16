import numpy as np


def perceptron(a, w, b=0):
    res = np.dot(a, w) + b
    print("res = {}".format(res))
    print("output = {}".format(True if res >= 0 else False))


perceptron([1, 1, 1], [1, 1, 1], -1)
