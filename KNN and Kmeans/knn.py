import math
import numpy as np
from typing import List
from collections import Counter
import matplotlib.pyplot as plt

from util import import_weather_dataset, normalize


def euclidean_distance(a, b) -> float:
    """Calculate the distance between two points. Returns the square of the distance (for performance reasons)."""
    n = len(a)
    if n != len(b):
        raise Exception("Input a and b do not have the same dimensions.")

    distance_sq = 0

    for i in range(n):
        distance_sq += (a[i] - b[i]) ** 2

    return distance_sq


def knn_get_most_common(labels, distances, k):
    """
    Returns the most common label for the given distances.
    When a shared first place is encountered this function will recursively call
    itself with k-1 until we have exactly one most commmon label.
    """
    results = []

    # Add the labels of the K nearest neighbors to a list.
    for i in range(k):
        results.append(labels[distances[i][0]])

    # Count how often each label appears in our result.
    counter = Counter(results)
    most_common = counter.most_common()

    # If we have a shared first place then we try again with K - 1
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return knn_get_most_common(labels, distances, k - 1)

    return most_common[0][0]


def knn(data, labels, point, k):
    """
    Classify a given 'point' using a given dataset.

    Parameters:
        data: input dataset
        labels: labels for the input dataset
        point: point to classify
        k: the K in KNN, aka how many nearest neighbors to look at

    Returns:
        label which (probably) belongs to the given point.
    """
    day_distances = []

    # Calculate the distance to all other points in the dataset.
    for i, other in enumerate(data):
        distance_sq = euclidean_distance(point, other)
        day_distances.append((i, distance_sq))

    # Sort the list, shortest distance first.
    day_distances.sort(key=lambda x: x[1])

    return knn_get_most_common(labels, day_distances, k)


def knn_k_tester(data, labels, test, test_labels):
    """
    Test which K value is the best for a given labeled dataset using a validation dataset.
    Also creates a plot.png in the working directory for the score/k.

    Parameters:
        data: input dataset
        labels: labels for the input dataset
        test: test/validation dataset
        test_labels labels for the test/validation dataset

    Returns:
        best K value for the given dataset/validation combo.
    """
    lst_k = list(range(1, 101))
    lst_score = []

    best_score = 0
    best_k = 0

    # Test all K's in the list
    for k in lst_k:
        score = 0
        total = len(test_data)

        # Classify each item in the validation set, and check if the answer is correct.
        for test_idx, test_entry in enumerate(test_data):
            res = knn(data, labels, test_entry, k)

            # print("Res = {}".format(res))
            # print("Expected = {}".format(test_labels[test_idx]))

            if res == test_labels[test_idx]:
                score = score + 1

        # Update best K if we have a new highscore
        if best_score < score:
            best_score = score
            best_k = k

        print("K = {}".format(k))
        print("Correct: {}".format(score))
        print("Total: {}\n".format(total))

        lst_score.append(score)

    print("Best K is {} with a score of {}".format(best_k, best_score))

    # Plot a line for k/score
    plt.plot(lst_k, lst_score)
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.ylim([30, 70])
    plt.savefig("knn.png")

    return best_k


if __name__ == "__main__":
    # Load input datasets
    data, labels = import_weather_dataset("KNN and Kmeans/dataset1.csv", 2000)
    test_data, test_labels = import_weather_dataset("KNN and Kmeans/validation1.csv", 2001)
    guess_data, _ = import_weather_dataset("KNN and Kmeans/days.csv", 2001)

    # Normalize all three input sets
    normalize(data)
    normalize(test_data)
    normalize(guess_data)

    # Get best K for a given data/validation combo.
    best_k = knn_k_tester(data, labels, test_data, test_labels)

    for i, guess in enumerate(guess_data):
        res = knn(data, labels, guess, best_k)
        print("Day {} - {}".format(i + 1, res))
