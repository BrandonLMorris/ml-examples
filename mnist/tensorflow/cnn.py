from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as pyplot


EPOCHS = 5000
MINIBATCH_SIZE = 50
SAVE_PATH = 'mnist-cnn-model.cpkt'

def cnn(x):
    """Definition of our model for a deep convolutional neural network

    This method serves as the main definition of our classifier model.
    :param x: The input image tensor
    :returns: A tuple that contains the output activations and the dropout
              probability used
    """
    relu = tf.nn.relu # shorthand
    keep_prob = tf.placeholder(tf.float32)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # Layer 1 (convolutional)
    W_conv1 = weights([5, 5, 1, 32]) # Output: 24x24x32
    b_conv1 = bias([32])
    h_conv1 = relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1)

    # Layer 2 (convolutional)
    W_conv2 = weights([5, 5, 32, 64]) # Output: 19x19x64
    b_conv2 = bias([64])
    h_conv2 = relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2)

    # Layer 3 (fully connected)
    W_fc1 = weights([7 * 7 * 64, 1024])
    b_fc1 = bias([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Apply dropout to prevent overfitting
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Last layer (fully connected)
    W_fc2 = weights([1024, 10])
    b_fc2 = bias([10])
    y_hat = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y_hat, keep_prob

def weights(shape):
    """Helper method; creates some randomly initialized weights"""
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias(shape):
    """Helper method; creates some randomly initialized biases"""
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    """Simplify our convolutional calls"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool(x):
    """Simplify our pooling calls"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def main(_):
    # Gather the data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Describe our model
    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])
    y_hat, keep_prob = cnn(x)

    # Measure error and optimize (describe the trianing procedure)
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_hat))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Actually run the training procedure (or load the pre-trained model)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if FLAGS.no_cache:
            for i in range(EPOCHS):
                # Train on minibatches of the data
                batch = mnist.train.next_batch(MINIBATCH_SIZE)

                # Mark our progress during training
                if i % 100 == 0:
                    print('Epoch {}: Accuracy {}%'.format(
                        i, accuracy.eval(feed_dict={
                            x:batch[0],
                            y_true:batch[1],
                            keep_prob:1.0
                        })))
                # Run the training step
                train_step.run(feed_dict={
                    x:batch[0], y_true:batch[1], keep_prob:0.5})
            saver.save(sess, SAVE_PATH)
        else:
            saver.restore(sess, SAVE_PATH)

        # Evaluate our acurracy
        test_acc = accuracy.eval(feed_dict={
            x:mnist.test.images, y_true:mnist.test.labels, keep_prob:1.0
        })
        print('Test accuracy is {:.2f}'.format(test_acc * 100))

        # Try out some adversarial examples
        img, label = mnist.train.next_batch(10)
        ae = fgsm(x, y_true, y_hat, 0.1).eval(feed_dict={
            x:img, y_true:label, keep_prob:1.0
        })
        ae_logits = y_hat.eval(feed_dict={x:ae, y_true:label, keep_prob:1.0})

        # Print out some examples
        for i in range(10):
            pic = ae[i].reshape([28, 28])
            pyplot.imshow(pic, cmap='gray')
            pyplot.title('Classified as {}'.format(tf.argmax(ae_logits, 1).eval()[i]))
            pyplot.show()


def fgsm(x, y_true, y_hat, epsilon=0.075):
    """Calculates the fast gradient sign method adversarial attack

    Following the FGSM algorithm, determines the gradient of the cost function
    wrt the input, then perturbs all the input in the direction that will cause
    the greatest error, with small magnitude.
    """
    loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_hat)
    grad, = tf.gradients(loss, x)
    scaled_grad = epsilon * tf.sign(grad)
    return tf.stop_gradient(x + scaled_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--no_cache', type=bool,
                        default=False,
                        help='Set if want to train model from scratch')
    FLAGS, unparsed = parser.parse_known_args()

    # If we don't have a saved versoin of the model, we'll have to train it
    if not os.path.exists(SAVE_PATH + '.index'):
        FLAGS.no_cache = True

    tf.app.run(main=main)
