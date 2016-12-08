#!/usr/local/env python3
"""
Simple k-means clustering example implementation. No optimization attempts
have been made, this is pure for example and educational purposes.

Data is randomly generated to form three clusters, the k-means is used to
learn the centers of these data clusters.
"""
import math
from matplotlib import pyplot
from random import randint
from numpy.random import normal

# Constant values for learning and data generation
SIGMA = 5
NUM_POINTS = 1500
EPSILON = 0.01
CENTER_1 = (33, 29)
CENTER_2 = (16, 85)
CENTER_3 = (77, 54)

def generate_noisy_data():
    """Creates the data set of clusters with some random noise"""
    noise = normal(0, SIGMA, NUM_POINTS * 2)
    X, Y = list(), list()
    noise_pos = 0
    for center in [CENTER_1, CENTER_2, CENTER_3]:
        for i in range(int(NUM_POINTS / (3 * 2))):
            X.append(center[0] + noise[noise_pos])
            noise_pos += 1
            Y.append(center[1] + noise[noise_pos])
            noise_pos += 1
    return X, Y


def extract_test_set(X, Y):
    """Segregates the data set into test and training sets (not used)"""
    test_size = round(0.3 * len(X))
    X_test, Y_test = list(), list()
    for i in range(test_size):
        index = randint(0, len(X) - 1)
        X_test.append(X.pop(index))
        Y_test.append(Y.pop(index))
    return X, X_test, Y, Y_test


def assign(X, Y, centers):
    """Assigns data points to the ceneter geometrically closest to it

    :param X: list of x-values from the data set
    :param Y: list of y-values form the data set
    :param centers: list of (x, y) tuples representing the coordinates of each
                    cluster center
    :returns: list of lists, each sub-list containing the (x, y) coordinates
              assigned to the cluster of that index (i.e. index 0 contains a
              list of coordinates assigned to the first center)
    """
    nearests = [list() for c in centers]
    for (x, y) in zip(X, Y):
        dists = [math.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (cx, cy) in centers]
        index = dists.index(min(dists))
        nearests[index].append((x, y))
    return nearests


def adjust(center, neighbors):
    """Calculates the mean coordinates of a list of (x, y) tuples

    Used to calculate the new position of a cluster center
    """
    if len(neighbors) == 0:
        return center
    # Watch out for overflow...
    avg_x = sum([n[0] for n in neighbors]) / len(neighbors)
    avg_y = sum([n[1] for n in neighbors]) / len(neighbors)
    return (avg_x, avg_y)


def main():
    # Plot the original data with defined center positions
    axis = [0, 100, 0, 100]
    X, Y = generate_noisy_data()
    X_train, X_test, Y_train, Y_test = extract_test_set(X, Y)
    for c in (CENTER_1, CENTER_2, CENTER_3):
        pyplot.plot(c[0], c[1], 'bo')
    pyplot.plot(X, Y, 'gx')
    pyplot.axis(axis)
    pyplot.title('Original Data Set')
    pyplot.show()

    # Initial center points randomly
    centers = [(randint(0, 100), randint(0, 100)) for i in range(3)]

    # Plot the initial positions on top of the training data
    for c in centers:
        pyplot.plot(c[0], c[1], 'ro')
    pyplot.plot(X_train, Y_train, 'gx')
    pyplot.axis(axis)
    pyplot.title('Initial Cluster Centers')
    pyplot.show()

    iterations = 0
    while True:
        # Move each center to the mean position of its assigned points
        neighbors = assign(X_train, Y_train, centers)
        new_centers = [adjust(c, n) for c, n in zip(centers, neighbors)]

        # Plot the movement of the centers
        pyplot.plot(X_train, Y_train, 'gx')
        for c in new_centers:
            pyplot.plot(c[0], c[1], 'ro')
        pyplot.axis(axis)
        pyplot.title('Iteration #{}'.format(iterations + 1))
        pyplot.show()

        # Stop if we've converged
        if all([abs(n[0] - c[0]) < EPSILON and abs(n[1] - c[1]) < EPSILON
                for (n, c) in zip(centers, new_centers)]):
            break
        else:
            centers = new_centers
            iterations += 1


if __name__ == '__main__':
    main()
