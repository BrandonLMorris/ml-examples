#!/usr/local/env python3
import math, sys, time, random, pickle
import numpy as np
from copy import deepcopy
from data_collection import get_training_set, print_image

SGD = False
GRAD_CHECK = False
TRAINING_SIZE = 1000
LEARNING_RATE = 1
LAMBDA = 1

# Constants for SGD
EPOCHS = 1000
BATCH_SIZE = 32

SAVE_FILE = 'thetas.txt'


def boolean_matrix_labels(labels):
    """Transform the scalar labels into a boolean list

    The list will have a 1 for the index that repesents the label, and zero
    elsewhere
    """
    results = list()
    for label in labels:
        l = [0 for i in range(10)]
        l[label] = 1
        results.append(l)
    return results


def logistic_sigmoid(z):
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        z = np.exp(z)
        return z / (1 + z)


def sigmoid_gradient(a):
    return [x * (1 - x) for x in a]


def cost_function(images, actuals, thetas):
    total = 0
    m = len(images)
    for image, actual in zip(images, actuals):
        predicted = forward_propogation(image, thetas[0], thetas[1])[-1]
        left = [y * np.log(p) for (p, y) in zip(predicted, actual)]
        right = [(1 - y) * np.log(1 - p) for (p, y) in zip(predicted, actual)]
        total += sum([x + y for (x, y) in zip(left, right)])
    total = - total / m

    # Calculate the regularization portion (sum of squares of parameters)
    reg = sum([sum([sum([c * c for c in r]) for r in t]) for t in thetas])
    return total + (LAMBDA / (2 * m)) * reg


def gradient_checking(images, actuals, thetas, gradients):
    epsilon = 1e-4
    for l in range(len(thetas)):
        for r in range(len(thetas[l])):
            for c in range(len(thetas[l][r])):
                plus = deepcopy(thetas)
                plus[l][r][c] += epsilon
                plus = cost_function(images, actuals, plus)
                minus = deepcopy(thetas)
                minus[l][r][c] -= epsilon
                minus = cost_function(images, actuals, minus)
                approx = (plus - minus) / (2 * epsilon)
                diff = abs(gradients[l][r][c] - approx)
                if diff > 1.0:
                    print('ERROR: Grad is off by {} for parameter [{}][{}][{}]'.format(diff, l, r, c))


def initialize_theta(rows, cols):
    epsilon = math.sqrt(6) / math.sqrt(cols + rows)
    results = list()
    for i in range(rows):
        row = [np.random.random() for j in range(cols)]
        row = [2 * epsilon * r - epsilon for r in row]
        results.append(row)
    return results


def forward_propogation(x, theta1, theta2):
    a1 = [1] + x
    z2 = [sum(a * b for (a, b) in zip(t, a1)) for t in theta1]
    a2 = [1] + [logistic_sigmoid(z) for z in z2]

    z3 = [sum(a * b for (a, b) in zip(t, a2)) for t in theta2]
    h = [logistic_sigmoid(z) for z in z3]
    return (a1, a2, h)


def backprop(activations, label, theta1, theta2):
    """Returns all the gradients for all the weights via backpropagation

    Note that this only computes the weight gradients given a *single*
    training example.

    :param activations: a tuple of all the activations (in order) resulting
                        from forward propogation
    :param label: a 10-element list of the actual picture label
    :param theta1: a 2D list of all the weights going from layer 1 to layer 2
    :param theta2: a 2D list of all the weights going from layer 2 to layer 3
    """
    a1, a2, a3 = activations
    delta3 = [a - y for (a, y) in zip(a3, label)]

    ### Not sure about this...
    # delta^l = ((Theta^l)' * delta^(l+1) .* a^l .* (1 - a^l)
    delta2 = [sum([a * b for (a, b) in zip(row, delta3)]) for row in theta2]
    delta2 = [d * a * (1. - a) for (d, a) in zip(delta2, a2)]
    ###

    acc = [list(), list()]
    acc[0] = [[0 for j in range(len(theta1[i]))] for i in range(len(theta1))]
    acc[1] = [[0 for j in range(len(theta2[i]))] for i in range(len(theta2))]
    for i in range(len(delta3)):
        for j in range(len(a2)):
            acc[1][i][j] += a2[j] * delta3[i]

    d2 = delta2[1:]
    for i in range(len(d2)):
        for j in range(len(a1)):
            acc[0][i][j] += a1[j] * d2[i]

    return acc


def main():
    sys.stdout.write('Gathering training data...')
    sys.stdout.flush()
    images, labels = get_training_set(TRAINING_SIZE)
    labels = boolean_matrix_labels(labels)
    sys.stdout.write('done!\n')

    theta1 = initialize_theta(25, 28 * 28 + 1)
    theta2 = initialize_theta(10, 26)

    if SGD:
        theta1, theta2 = sgd_training(images, labels, theta1, theta2)
    else:
        theta1, theta2 = gd_training(images, labels, theta1, theta2)

    # See how well we did
    prediction = forward_propogation(images[0], theta1, theta2)[-1]
    print(prediction)
    prediction = prediction.index(max(prediction))
    actual = labels[0].index(1)
    print('predicted: {} actually: {}'.format(prediction, actual))

    correct, incorrect = 0, 0
    for (image, label) in list(zip(images, labels)):
        prediction = forward_propogation(image, theta1, theta2)[-1]
        prediction = prediction.index(max(prediction))
        actual = label.index(1)
        if (prediction == actual):
            correct += 1
        else:
            incorrect += 1
    print('for the *training* set:')
    print('{} correct, {} incorrect'.format(correct, incorrect))
    print('accuracy of {}'.format(correct / (correct + incorrect)))


def save_thetas(thetas):
    with open(SAVE_FILE, 'wb') as f:
        pickle.dump(thetas, f)

def read_thetas(thetas):
    with open(SAVE_FILE, 'rb') as f:
        return pickle.load(f)


def gd_training(images, labels, theta1, theta2):
    prev_cost = 2 << 32 # Arbitrarily large number
    print('Starting (non-stochastic) GD training:')
    print('  {} images'.format(TRAINING_SIZE))
    print('  {} learning rate\n'.format(LEARNING_RATE))
    start_time = time.time()
    zipped = list(zip(images, labels))
    epoch = -1
    while True:
        epoch += 1
        if epoch % 100 == 0: print('--- Starting epoch {} ---'.format(epoch))
        grad_total = None
        for image, label in zipped:
            activations = forward_propogation(image, theta1, theta2)
            grad = backprop(activations, label, theta1, theta2)
            if not grad_total:
                grad_total = grad
            else:
                for layer in range(len(grad_total)):
                    for row in range(len(grad_total[layer])):
                        for col in range(len(grad_total[layer][row])):
                            grad_total[layer][row][col] += grad[layer][row][col]

        # Regularize the gradients
        thetas = [theta1, theta2]
        for layer in range(len(grad_total)):
            for row in range(len(grad_total[layer])):
                for col in range(len(grad_total[layer][row])):
                    if col > 0:
                        grad_total[layer][row][col] = (1 / TRAINING_SIZE) * (grad_total[layer][row][col] + (LAMBDA * thetas[layer][row][col]))
                    else:
                        grad_total[layer][row][col] = (1 / TRAINING_SIZE) * grad_total[layer][row][col]


        # Update the weights with gradient descent
        for layer in range(len(grad_total)):
            for row in range(len(grad_total[layer])):
                for col in range(len(grad_total[layer][row])):
                    thetas[layer][row][col] -= LEARNING_RATE * grad_total[layer][row][col]
        theta1, theta2 = thetas

        # Print out the cost after every 10 updates
        if epoch % 10 == 0:
            cost = cost_function(images, labels, [theta1, theta2])
            print('Cost at epoch {}: {}'.format(epoch, cost))

            # Define convergence as cost changing by less than 0.01%
            # diff = abs(prev_cost - cost)
            # perc = diff / prev_cost
            # FIXME if perc * 100 < 0.01: break
            if cost < 2.1: break

        if epoch % 100 == 0:
            # Save the current values of the parameters in a file
            save_thetas(thetas)

    min, sec = divmod(time.time() - start_time, 60)
    print('Training time: {} minutes, {} seconds'.format(min, int(sec)))
    return theta1, theta2


def sgd_training(images, labels, theta1, theta2):
    print('Starting training:')
    print('  {} images'.format(TRAINING_SIZE))
    print('  {} epochs'.format(EPOCHS))
    print('  {} images per minibatch'.format(BATCH_SIZE))
    print('  {} learning rate\n'.format(LEARNING_RATE))
    start_time = time.time()
    zipped = list(zip(images, labels))
    for epoch in range(EPOCHS):
        if epoch % 100 == 0: print('--- Starting epoch {} ---'.format(epoch))
        grad_total = None
        batch = [random.choice(zipped) for i in range (BATCH_SIZE)]
        predictions = list()
        for image, label in batch:
            activations = forward_propogation(image, theta1, theta2)
            predictions.append(activations[-1])
            grad = backprop(activations, label, theta1, theta2)
            if not grad_total:
                grad_total = grad
            else:
                for layer in range(len(grad_total)):
                    for row in range(len(grad_total[layer])):
                        for col in range(len(grad_total[layer][row])):
                            grad_total[layer][row][col] += grad[layer][row][col]

        # Average the gradients
        thetas = [theta1, theta2]
        for layer in range(len(grad_total)):
            for row in range(len(grad_total[layer])):
                for col in range(len(grad_total[layer][row])):
                    if col > 0:
                        grad_total[layer][row][col] = (1 / BATCH_SIZE) * (grad_total[layer][row][col] + thetas[layer][row][col])
                    else:
                        grad_total[layer][row][col] = (1 / BATCH_SIZE) * grad_total[layer][row][col]

        # Goodness I hope this works
        labels = [b[1] for b in batch]
        images = [b[0] for b in batch]
        if GRAD_CHECK and epoch == 0:
            gradient_checking(images, labels, thetas, grad_total)


        # Update the weights with gradient descent
        for layer in range(len(grad_total)):
            for row in range(len(grad_total[layer])):
                for col in range(len(grad_total[layer][row])):
                    thetas[layer][row][col] -= LEARNING_RATE * grad_total[layer][row][col]
        theta1, theta2 = thetas

        # Print out the cost of this minibatch
        if epoch % 50 == 0:
            cost = cost_function(images, labels, [theta1, theta2])
            print('Cost at epoch {}: {}'.format(epoch, cost))

    min, sec = divmod(time.time() - start_time, 60)
    print('Training time: {} minutes, {} seconds'.format(min, int(sec)))
    return theta1, theta2



if __name__ == '__main__':
    main()
