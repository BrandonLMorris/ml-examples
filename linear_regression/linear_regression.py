#!/usr/local/env python3
"""
Simple linear regression with gradient descent. No optimization attempts have
been made, this is purely for example and educational purposes.

Data is randomly generated and saved to a file in the current directory for
reproduceability. To regenerate the data, simply delete the
'linear_regression_data.txt' file.
"""

from random import randint
from numpy import linspace, random, array
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression

# Defined parameters for linear function (y = ax + b)
A = 4
B = 15
SIGMA = 5
LEARNING_RATE = .0003
DATA_FILE = 'linear_regression_data.txt'


def generate_noisy_data(slope, intercept, start=0, stop=100):
    """Create some linear data with some noise added to it"""
    num_points = abs(start) + abs(stop)
    line = linspace(start, stop, num_points)
    noise = random.normal(0, SIGMA, num_points)
    noisy = [(slope * d + intercept) + n for (d, n) in zip(line, noise)]
    return noisy


def save_data(x, y):
    """Save the x and y coordinates to a file"""
    with open(DATA_FILE, 'w+') as f:
        for (xx, yy) in zip(x, y):
            f.write('{} {}\n'.format(xx, yy))


def get_data():
    """Retrieve the data from the cached file

    If the data can't be found in a saved file, generate some new data
    and save that to a file
    """
    x, y = list(), list()
    try:
        with open(DATA_FILE, 'r') as f:
            lines = f.readlines()
            for line in lines:
                xx, yy = line.split()
                x.append(float(xx))
                y.append(float(yy))
        return x, y
    except FileNotFoundError:
        # No data file, generate the data
        y = generate_noisy_data(A, B)
        x = list(range(len(y)))
        save_data(x, y)
        return get_data()


def hypothesis(theta0, theta1, x):
    """Return our hypothesis, or guess, given the parameters and input value"""
    return theta0 + (theta1 * x)


def cost(X, Y, theta0, theta1):
    """Find the total cost of our model from the training examples

    :param X: a list of input values
    :param Y: a list of ouput values
    :param theta0: the value of the first parameter of the model
    :param theta1: the value of the second parameter of the model
    """
    errors = [(hypothesis(theta0, theta1, x) - y) for (x, y) in zip(X, Y)]
    return (1 / (2 * len(Y))) * sum([e * e for e in errors])


def descend(alpha, theta0, theta1, X, Y):
    """One iteration of gradient descent

    Adjusts the model to reflect a single step of the linear descent algorithm

    :param alpha: the learning rate
    :param thetas: a list of parameters for the model
    :param Y: a list of output values
    :param X: a list of input values
    """
    results = list()
    n = len(Y)
    partial0 = (1 / n) * sum([(hypothesis(theta0, theta1, x) - y)
                              for (x, y) in zip(X, Y)])
    partial1 = (1 / n) * sum([(hypothesis(theta0, theta1, x) - y) * x
                              for (x, y) in zip(X, Y)])
    new_theta0 = theta0 - alpha * partial0
    new_theta1 = theta1 - alpha * partial1
    return (new_theta0, new_theta1)


def r_squared(Y_predict, Y_true):
    """Calculate the R-squared value of the model

    :param theta0: the first parameter of the model
    :param theta1: the second parameter of the model
    :param X: a list of input values
    :param Y: a list of output values
    """
    u = sum([(yt - yp) ** 2 for (yt, yp) in zip(Y_true, Y_predict)])
    mean = sum(Y_true) / len(Y_true)
    v = sum([(y - mean) ** 2 for y in Y_true])
    return (1 - (u / v))


def extract_test_set(X, Y):
    """Segregate the data set into test and training sets"""
    num_test = round(len(X) * .3)
    X_test, Y_test = list(), list()
    for i in range(num_test):
        index = randint(0, len(X) - 1)
        X_test.append(X.pop(index))
        Y_test.append(Y.pop(index))
    return (X, X_test, Y, Y_test)


def main():
    # Plot the original data set
    X, Y = get_data()
    pyplot.title("Original Data Set")
    pyplot.plot(Y, 'k.')
    pyplot.show()

    # Create our initial values
    X_train, X_test, Y_train, Y_test = extract_test_set(X, Y)
    theta0, theta1 = 0, 0

    # Train out model
    prev_cost = 2 << 32 # Arbitrarily large number
    iterations = 0
    while True:
        theta0, theta1 = descend(LEARNING_RATE, theta0, theta1, X_train,
                                 Y_train)
        current_cost = cost(X_train, Y_train, theta0, theta1)

        # Stop if we've converged
        if abs(current_cost - prev_cost) < 0.0001:
            print('{} iterations'.format(iterations))
            break
        else:
            iterations += 1
            prev_cost = current_cost

    # Plot our results
    result = [hypothesis(theta0, theta1, yy) for yy in range(0, 100)]
    pyplot.title("By-Hand Results")
    pyplot.plot(X, Y, 'k.')
    pyplot.plot(result)
    pyplot.show()

    Y_predict = [hypothesis(theta0, theta1, x) for x in X_test]
    print('R^2 from by-hand: {}'.format(r_squared(Y_predict, Y_test)))

    # Same algorithm, but utilize sklearn
    lr = LinearRegression()
    X_vector, Y_vector = (array(X_train).reshape(-1, 1),
                          array(Y_train).reshape(-1, 1))
    lr.fit(X_vector, Y_vector)
    sk_predictions = [lr.predict(x) for x in X]
    print('R^2 from sklearn: {}'.format(lr.score(X_vector, Y_vector)))

    # Plot the results
    sk_y_predict = [lr.predict(x)[0, 0] for x in X]
    pyplot.title('sklearn Results')
    pyplot.plot(X, Y, 'k.')
    pyplot.plot(X, sk_y_predict)
    pyplot.show()


if __name__ == '__main__':
    main()

