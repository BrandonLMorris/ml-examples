from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import argparse

TRAINING_EPOCHS = 50000
MINIBATCH_SIZE = 50
SAVE_PATH = '/tmp/mnist-feedforward-model.cpkt'

def neural_network(x):
    """Feedforward shallow neural network
    """
    L_in, L1, L2, L_out = 784, 200, 25, 10

    # First layer: 784 -> 200
    x_image = tf.reshape(x, [-1, L_in])
    W1 = weight_var([L_in, L1])
    b1 = bias_var([L1])
    a1 = tf.nn.relu(tf.matmul(x_image, W1) + b1)

    # Apply dropout to our activations
    keep_prob = tf.placeholder(tf.float32)
    a1_drop = tf.nn.dropout(a1, keep_prob)

    # Second layer: 200 -> 25
    W2 = weight_var([L1, L2])
    b2 = bias_var([L2])
    a2 = tf.nn.relu(tf.matmul(a1_drop, W2) + b2)
    a2_drop = tf.nn.dropout(a2, keep_prob)

    # Output layer: 25 -> 10
    W3 = weight_var([L2, L_out])
    b3 = bias_var([L_out])
    a3 = tf.nn.relu(tf.matmul(a2_drop, W3) + b3)

    return a3, keep_prob


def weight_var(shape):
    '''Wrapper for making a weight variable'''
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_var(shape):
    '''Wrapper for making a bias constant'''
    return tf.Variable(tf.constant(0.1, shape=shape))


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])
    y_hat, keep_prob = neural_network(x)

    # Evaluate/optimize our model
    cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_true, logits=y_hat))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.no_cache:
            for i in range(TRAINING_EPOCHS):
                batch = mnist.train.next_batch(MINIBATCH_SIZE)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x:batch[0], y_true: batch[1], keep_prob: 1.0
                    })
                    print('Training accuracy ({}/{}): {}'.format(
                        i, TRAINING_EPOCHS, train_accuracy))
                train_step.run(feed_dict={
                    x: batch[0],
                    y_true: batch[1],
                    keep_prob: 0.5
                })
            saver.save(sess, SAVE_PATH)
        else:
            saver.restore(sess, SAVE_PATH)

        # Test our trained model
        print('Test accuracy: {}'.format(accuracy.eval(feed_dict={
            x: mnist.test.images,
            y_true: mnist.test.labels,
            keep_prob: 1.0
        })))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--no_cache', type=bool,
                        default=False,
                        help='Set if want to train model from scratch')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)
