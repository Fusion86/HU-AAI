import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from collections import Counter

from util import import_weather_dataset, normalize


class Solution:
    def __init__(self, score, centroids, clusters):
        self.score = score
        self.centroids = centroids
        self.clusters = clusters

    def __repr__(self):
        return "<Solution, score: {}>".format(self.score)


def euclidean_distance(a, b) -> float:
    """Calculate the distance between two points. Returns the square of the distance (for performance reasons)."""
    n = len(a)
    if n != len(b):
        raise Exception("Input a and b do not have the same dimensions.")

    distance_sq = 0

    for i in range(n):
        distance_sq += (a[i] - b[i]) ** 2

    return distance_sq


def kmeans(k: int, data, initial_centroids=None, iteration: int = 0):
    if initial_centroids == None:
        # Pick initial centroids.
        centroids = random.sample(list(data), k)
    else:
        centroids = initial_centroids.copy()

    clusters = [[] for _ in range(k)]

    for point in data:
        # Find nearest centroid.
        min_distance = float("inf")
        closest_centroid_idx = None

        for centroid_idx, centroid in enumerate(centroids):
            distance_sq = euclidean_distance(point, centroid)
            if distance_sq < min_distance:
                min_distance = distance_sq
                closest_centroid_idx = centroid_idx

        # Add this point to the cluster belonging to the nearest centroid.
        clusters[closest_centroid_idx].append(point)

    # Calculate new centroids based on the new clusters we created.
    for cluster_idx, cluster in enumerate(clusters):
        new_centroid = np.mean(cluster, 0)
        centroids[cluster_idx] = new_centroid

    # We are done when the centroids don't change.
    # TODO: We might also be done when the clusters don't change?
    if initial_centroids != None and np.array_equal(initial_centroids, centroids):
        return (centroids, clusters)

    # We aren't done yet, run again with the new centroids.
    return kmeans(k, data, centroids, iteration + 1)


# TODO: Not sure if this is correct
def kmeans_intra_cluster_distance(centroids, clusters):
    distances = []

    for i, cluster in enumerate(clusters):
        centroid = centroids[i]

        for point in cluster:
            distance = euclidean_distance(centroid, point)
            distances.append(distance)

    return np.mean(distances)


def np_indexof_2d(array, row_to_find):
    for i, row in enumerate(array):
        if np.array_equal(row, row_to_find):
            return i


def print_clusters_info(data, labels, clusters):
    for i, cluster in enumerate(clusters):
        cluster_labels = []

        for point in cluster:
            # Find which label belongs to a point
            idx = np_indexof_2d(data, point)
            label = labels[idx]
            cluster_labels.append(label)

        counter = Counter(cluster_labels)

        print("Cluster {} ({})".format(i + 1, counter.most_common(1)[0][0]))
        for label in counter:
            print("  {}: {}".format(label, counter[label]))


if __name__ == "__main__":
    # Set random seed
    random.seed("2151901553968352745")

    # Load input dataset
    data, labels = import_weather_dataset("dataset1.csv", 2000)

    # Normalize dataset
    normalize(data)

    lst_k = list(range(2, 13))
    lst_scores = []

    for k in lst_k:
        print("Clustering where K is {}".format(k), end="")
        solutions: List[Solution] = []

        # Run K-means 10 times and select the best cluster.
        for i in range(10):
            # TODO: Cluster vote meaning
            centroids, clusters = kmeans(k, data)
            score = kmeans_intra_cluster_distance(centroids, clusters)
            solutions.append(Solution(score, centroids, clusters))
            print(".", end="")
        print()  # Newline

        # Pick best solution (where the score/distance is the lowest)
        solutions.sort(key=lambda x: x.score)
        lst_scores.append(solutions[0].score)

        # Print best solution info
        print_clusters_info(data, labels, solutions[0].clusters)
        print()

    # Plot
    plt.plot(lst_k, lst_scores)
    plt.xticks(lst_k)  # Hacky
    plt.xlabel("K")
    plt.ylabel("Variance")
    plt.savefig("kmeans.png")
