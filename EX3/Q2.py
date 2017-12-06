from optparse import OptionParser

import matplotlib
import numpy
import numpy.random
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn import preprocessing
import sklearn


matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    # binary classification of a digit as either 0 or 8
    plus, minus = 0, 8
    train_perm = numpy.where((labels[:60000] == plus) | (labels[:60000] == minus))[0]
    train_idx = numpy.random.RandomState(0).permutation(train_perm)
    test_perm = numpy.where((labels[60000:] == plus) | (labels[60000:] == minus))[0]
    test_idx = numpy.random.RandomState(0).permutation(test_perm) + 60000

    train_data_unscaled = data[train_idx, :].astype(float)
    train_labels = (labels[train_idx] == (plus or minus)) * 2 - 1

    test_data_unscaled = data[test_idx, :].astype(float)
    test_labels = (labels[test_idx] == (plus or minus)) * 2 - 1

    # mean-center each coordinate of our data points
    train_data = sklearn.preprocessing.scale(train_data_unscaled, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, with_std=False)

    # normalize to unit vectors
    train_normalized = preprocessing.normalize(train_data, norm='l2')
    test_normalized = preprocessing.normalize(test_data, norm='l2')

    return train_normalized, train_labels, test_normalized, test_labels


def perceptron(x, y, w):
    """implements a step of the perceptron algorithm, the weight vector w"""
    prediction = (numpy.inner(w, x) >= 0)*2 - 1
    if prediction != y:
        w += y * x

    return w


def perceptron_train(xs, ys):
    w = numpy.zeros(784)
    for x, y in zip(xs, ys):
        w = perceptron(x, y, w)
    return w


def accuracy_perceptron(test_data, test_labels, w):
    success = 0
    for x, y in zip(test_data, test_labels):
        prediction = (numpy.inner(w, x) >= 0)*2 - 1
        if prediction == y:
            success += 1

    # Accuracy of the prediction
    success_rate = (float(success) / len(test_data))
    return success_rate


def part_a():
    train_data, train_labels, test_data, test_labels = load_data()
    n_range = [5, 10, 50, 100, 500, 1000, 5000]
    # noinspection PyPep8Naming
    T_range = xrange(0, 100)
    accuracies = []

    for n in n_range:
        xs = train_data[:n]
        ys = train_labels[:n]
        accuracy = []
        for _ in T_range:
            # different random order of inputs
            permutation = numpy.random.permutation(xrange(n))
            xs = xs[permutation]
            ys = ys[permutation]
            w = perceptron_train(xs, ys)
            accuracy.append(accuracy_perceptron(test_data, test_labels, w))
        accuracies.append(accuracy)

    # the 5% and 95% percentiles of the accuracies obtained
    means = numpy.mean(accuracies, axis=1).reshape(-1, 1)
    percentiles = numpy.percentile(accuracies, [5, 95], axis=1).T
    columns = ['mean', '5% percentile', '95% percentile']
    df = pd.DataFrame(numpy.concatenate((means, percentiles), axis=1), n_range, columns)
    df.index.name = 'n'
    print df


def part_b():
    train_data, train_labels, test_data, test_labels = load_data()
    w = perceptron_train(train_data, train_labels)
    plt.imshow(numpy.reshape(w, (28, 28)), interpolation='nearest')
    plt.title('w as a picture')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 2(b)')
    plt.savefig('q2_part_b.png')


def part_c():
    train_data, train_labels, test_data, test_labels = load_data()
    w = perceptron_train(train_data, train_labels)
    accuracy = accuracy_perceptron(test_data, test_labels, w)
    print 'The accuracy while training on the whole set is: {}'.format(accuracy)


def part_d():
    train_data, train_labels, test_data, test_labels = load_data()
    w = perceptron_train(train_data, train_labels)

    failures = 0
    for x, y in zip(test_data, test_labels):
        prediction = (numpy.inner(w, x) >= 0) * 2 - 1
        if prediction != y:
            failures += 1
            plt.imshow(numpy.reshape(x, (28, 28)), interpolation='nearest')
            plt.title('x as a picture')
            fig = plt.gcf()
            fig.canvas.set_window_title('Programming Assignment: Question 2(b)')
            plt.savefig('q2_part_c_{}.png'.format(failures))
        if failures == 2:
            break


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--a", action='store_true', help="Run question 2 part (a)")
    parser.add_option("--b", action='store_true', help="Run question 2 part (b)")
    parser.add_option("--c", action='store_true', help="Run question 2 part (c)")
    parser.add_option("--d", action='store_true', help="Run question 2 part (d)")
    (options, args) = parser.parse_args()

    if options.a:
        print 'Running Question 2 part (a)...'
        part_a()
    elif options.b:
        print 'Running Question 2 part (b)...'
        part_b()
    elif options.c:
        print 'Running Question 2 part (c)...'
        part_c()
    elif options.d:
        print 'Running Question 2 part (d)...'
        part_d()
    else:
        print 'Error: please run \'--help\' to see options'
