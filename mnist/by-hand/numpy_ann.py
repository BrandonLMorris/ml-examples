import sys
import time
import numpy as np
import matplotlib as plt
from data_collection import get_training_set_array, get_test_set_array, print_image

TRAINING_SIZE = 1000
TEST_SIZE = 500
ITER_INTERVAL = 300

# Hyperparameters of the model/training
CONVERGENCE = 0.003
LEARNING_RATE = .03
LAMBDA = 1


def main():
    sys.stdout.write('Gathering training data...')
    sys.stdout.flush()
    images, labels = get_training_set_array(TRAINING_SIZE)
    labels = boolean_matrix_labels(labels)
    print('done!')

    thetas = initialize_theta(25, 28 * 28 + 1), initialize_theta(10, 26)

    start = time.time()
    thetas = gd_training(images, labels, thetas)
    mi, sec = divmod(time.time() - start, 60)
    print('Training took {} minutes, {} seconds'.format(int(mi), int(sec)))

    # Training accuracy
    actuals = np.array([np.where(l == 1)[0][0] for l in labels])
    correct = evaluate(images, actuals, thetas)
    print('{} correct out of {}: {}% accuracy'.format(
            correct, TRAINING_SIZE, 100 * correct / TRAINING_SIZE))

    # Testing accuracy
    sys.stdout.write('\nGathering test data...')
    sys.stdout.flush()
    test_images, test_labels = get_test_set_array(TEST_SIZE)
    print('done!')
    correct = evaluate(test_images, test_labels, thetas)
    print('{} correct out of {}: {}% accuracy'.format(
            correct, 100 * TEST_SIZE, correct / TEST_SIZE))

    view_sample(10, test_images, test_labels, thetas)


def view_sample(num, images, labels, thetas):
    """Visually see a sample of the test predictions with matplotlib

    :param num: the number of examples to view
    """
    fmt_str = '#{} of {}: Predicted {}, actual {}'
    img_len = images.shape[1]
    for i in range(num):
        x = int(TEST_SIZE * np.random.random())
        image, label = images[x,:], labels[x]
        h = forward_prop(image.reshape(1, img_len), thetas)[-1]
        predict = np.where(h == h.max())[1][0]
        print_image(image, fmt_str.format(i+1, num, predict, label))


def gd_training(images, labels, thetas):
    """Train the parameters with standard gradient descent

    :param images: a numpy array of the training images
    :param labels: a numpy boolean matrix of the training labels
    :param thetas: the parameters for the model
    :return: the trained parameters for the model
    """
    prev_cost = 2 << 31 # Arbitrary large number
    iteration = -1
    while True:
        if iteration % ITER_INTERVAL == 0:
            cst = cost_function(images, labels, thetas)
            print('Starting iteration {}: cost of {}'.format(iteration, cst))

            # Check for convergence
            if abs(cst - prev_cost) < CONVERGENCE:
                break
            else:
                prev_cost = cst

        activations = forward_prop(images, thetas)
        gradients = backprop(activations, labels, thetas)

        # Adjust the weights with gradient descent
        # thetas = (thetas[0] - LEARNING_RATE * gradients[0],
        #           thetas[1] - LEARNING_RATE * gradients[1])
        thetas = [t - LEARNING_RATE * g for t, g in zip(thetas, gradients)]
        iteration += 1
    return thetas


def evaluate(images, labels, thetas):
    """Evaluate the model with a set of images and labels

    :param images: a numpy array of images to evaluate against
    :param labels: the set of the lables corresponding to the images
    :param thetas: a list of the parameters for the network
    :return: the number of correct predictions made by the model with these
             parameters
    """
    predictions = forward_prop(images, thetas)[-1]
    predicteds = [np.where(r == r.max())[0][0] for r in predictions]
    actuals = labels
    correct = 0
    for p, a in zip(predicteds, actuals):
        if p == a: correct += 1
    return correct


def boolean_matrix_labels(labels):
    """Transform the scalar labels into a boolean list

    The list will have a 1 for the index that repesents the label, and zero
    elsewhere, i.e. 4 => [0 0 0 0 1 0 0 0 0 0].
    """
    results = list()
    for label in labels:
        l = [0 for i in range(10)]
        l[label] = 1
        results.append(l)
    return np.array(results)


def cost_function(images, labels, thetas):
    """Determine the cost of the model, with regularization"""
    m = images.shape[0]
    h = forward_prop(images, thetas)[-1]
    left = -labels * np.log(h)
    right = (1 - labels) * np.log(1 - h)
    unreged = (1 / m) * (left - right).sum()

    t1_reg = (thetas[0][:, 1:] * thetas[0][:, 1:]).sum()
    t2_reg = (thetas[1][:, 1:] * thetas[1][:, 1:]).sum()
    reg = (LAMBDA / (2 * m)) * (t1_reg + t2_reg)
    return unreged + reg


def sigmoid(z):
    """Calculate the logistic sigmoid function"""
    return 1 / (1 + np.exp(-z))


def forward_prop(images, thetas):
    """Calculate a forward pass of the images through the model

    :return: the activations from each layer
    """
    theta1, theta2 = thetas
    m = images.shape[0]
    a1 = np.c_[np.ones(m), images]
    a2 = np.c_[np.ones(m), sigmoid(a1.dot(theta1.T))]
    h = sigmoid(a2.dot(theta2.T))
    return a1, a2, h


def backprop(activations, labels, thetas):
    """Calculate the parameter gradients via backpropagation

    :param activations: the set of activation values of each layer from
                        forward forward propagation
    :param labels: the labels associated with the images
    :param thetas: the parameters of the model
    :return: the gradients of each parameter
    """
    theta1, theta2 = thetas
    a1, a2, h = activations
    m = h.shape[0]
    delta3 = h - labels
    delta2_nograd = theta2.T.dot(delta3.T)
    delta2 = delta2_nograd.T * a2 * (1 - a2)
    grad2 = (1 / m) * delta3.T.dot(a2)
    grad1 = (1 / m) * delta2.T.dot(a1)

    # Add the regularization (except j=0)
    reg2 = np.c_[np.zeros(theta2.shape[0]), (LAMBDA / m) * theta2[:, 1:]]
    reg1 = np.c_[np.zeros(theta1.shape[0]), (LAMBDA / m) * theta1[:, 1:]]

    grad2 += reg2
    grad1 = grad1[1:, :] + reg1

    return grad1, grad2


def initialize_theta(rows, cols):
    """Randomly initialize theta to some starting values.

    The initialization takes into account the layer size for better
    initialization.
    """
    epsilon = np.sqrt(6) / np.sqrt(cols + rows)
    results = list()
    for i in range(rows):
        row = np.random.random(cols)
        row = 2 * epsilon * row - epsilon
        results.append(row)
    return np.array(results)


if __name__ == '__main__':
    main()

