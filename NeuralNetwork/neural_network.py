import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from neuron import NeuralNetwork


def normalize(data):
    """Normalizes a given array. Edits the array in place, aka does not return."""

    max_values = np.max(data, 0)
    min_values = np.min(data, 0)

    for row in data:
        for col in range(data.shape[1]):
            row[col] = (row[col] - min_values[col]) / (
                max_values[col] - min_values[col]
            )


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    data = np.genfromtxt("iris.data", delimiter=",", usecols=[0, 1, 2, 3])
    labels = np.genfromtxt("iris.data", delimiter=",", usecols=[4], dtype=str)
    output_labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    # normalize(data)

    # Randomize the data order
    data_label_pairs = []
    for i in range(len(data)):
        data_label_pairs.append((data[i], labels[i]))
    random.shuffle(data_label_pairs)

    # Split the dataset in two, one for training and one for testing
    train_data = data_label_pairs[:100]
    test_data = data_label_pairs[100:]

    network = NeuralNetwork([4, [4, 4, 4], 3])

    print("Network structure:")
    print(network.network_structure_str)

    learning_factor = 1 / len(output_labels)
    step = 10
    total_iterations = 1000

    x_data = []
    training_scores = []
    test_scores = []

    for iterations in range(0, total_iterations, step):
        # Train network
        print(
            "\nTraining in {} iterations with learning factor {}.".format(
                iterations, learning_factor
            )
        )
        network.train(
            train_data,
            output_labels,
            step,
            learning_factor,
        )

        print("\nAfter training")
        # Possible improvement would be to use the cost function while training
        training_score = network.validate(train_data, output_labels)
        test_score = network.validate(test_data, output_labels)

        print("Training Score: {}".format(training_score))
        print("Test Score: {}".format(test_score))

        x_data.append(iterations)
        training_scores.append(training_score)
        test_scores.append(test_score)

    plt.suptitle("Learning MNIST (learning factor = {:.2g})".format(learning_factor))
    plt.title("Network structure: {}".format(network.network_structure_str))
    plt.plot(x_data, training_scores, label="Training Score")
    plt.plot(x_data, test_scores, label="Test Score")
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(
        "images/learning_mnist_{}_{:.3g}.png".format(
            network.network_structure_str, learning_factor
        )
    )
