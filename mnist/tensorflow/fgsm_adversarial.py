from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import argparse
import matplotlib.pyplot as pyplot
import random

from feedforward_ann import neural_network, SAVE_PATH

def fgsm(x, y_true, y_hat, epsilon=0.075):
    loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_hat)
    grad, = tf.gradients(loss, x)
    scaled_grad = epsilon * tf.sign(grad)
    return tf.stop_gradient(x + scaled_grad)

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    # Build model
    x = tf.placeholder(tf.float32, [None, 784])
    y_true = tf.placeholder(tf.float32, [None, 10])
    y_hat, keep_prob = neural_network(x)
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_true, 1))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, SAVE_PATH)
        img, label = mnist.test.images, mnist.test.labels

        # Create and then test our adversarial examples
        ae = fgsm(x, y_true, y_hat).eval(feed_dict={
            x:img, y_true:label, keep_prob:1.0})
        ae_logits = y_hat.eval(feed_dict={
            x:ae, y_true:label, keep_prob:1.0})

        # Calculate the fooling success rate
        fooled = tf.not_equal(tf.argmax(ae_logits, 1), tf.argmax(label, 1))
        fooling_acc = tf.reduce_mean(tf.cast(fooled, tf.float32))
        print('Fooling success rate: {:.2f}%'.format(fooling_acc.eval()*100))

        # Display a random adversarial example (may not be successful)
        index = random.randint(0, len(img))
        pic = img[index].reshape([28, 28])
        pyplot.imshow(pic, cmap='gray')
        pyplot.title('Classified as {}'.format(tf.argmax(label, 1).eval()[index]))
        pyplot.show()

        pic = ae[index].reshape([28, 28])
        pyplot.imshow(pic, cmap='gray')
        pyplot.title('Classified as {}'.format(tf.argmax(ae_logits, 1).eval()[index]))
        pyplot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main)
