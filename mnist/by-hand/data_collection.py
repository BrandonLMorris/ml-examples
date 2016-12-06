import os
import sys
import numpy as np
import matplotlib.pyplot as plt

DATA_FOLDER = '/Users/bmorris/Documents/MNIST Dataset'
TRAINING_IMAGES = os.path.join(DATA_FOLDER, 'train-images-idx3-ubyte')
TRAINING_LABELS = os.path.join(DATA_FOLDER, 'train-labels-idx1-ubyte')
TEST_IMAGES = os.path.join(DATA_FOLDER, 't10k-images-idx3-ubyte')
TEST_LABELS = os.path.join(DATA_FOLDER, 't10k-labels-idx1-ubyte')


def get_training_set_array(num=None):
    images, labels = get_training_set(num)
    return np.array(images), np.array(labels)


def get_test_set_array(num=None):
    images, labels = get_test_set(num)
    return np.array(images), np.array(labels)


def get_training_set(num=None):
    images = read_images(TRAINING_IMAGES, num)
    labels = read_labels(TRAINING_LABELS, num)
    return images, labels


def get_test_set(num=None):
    images = read_images(TEST_IMAGES, num)
    labels = read_labels(TEST_LABELS, num)
    return images, labels


def read_images(filename, num=None):
    images = list()
    try:
        with open(filename, 'rb') as f:
            f.read(4) # skip the magic number
            num_images = int.from_bytes(f.read(4), 'big')
            f.read(8) # skip dimensions (assumed to be 28x28)
            for i in range(num or num_images):
                image = list()
                for j in range(28 * 28):
                    image.append(int.from_bytes(f.read(1), 'big'))
                images.append(image)
    except FileNotFoundError:
        print('Hm, I couldn\'t find the any images at {}'.format(filename))
        sys.exit(1)
    return images


def read_labels(filename, num=None):
    try:
        with open(filename, 'rb') as f:
            f.read(4) # skip the magic number
            num_labels = int.from_bytes(f.read(4), 'big')
            labels = [int.from_bytes(f.read(1), 'big')
                      for i in range(num or num_labels)]
    except FileNotFoundError:
        print('Hm, I couldn\'t find any labels at {}'.format(filename))
        sys.exit(1)
    return labels


def print_image(pixels, title=''):
    picture = np.array(pixels, dtype='uint8').reshape((28, 28))
    plt.imshow(picture, cmap='gray')
    plt.title(title)
    plt.show()

