from optparse import OptionParser

import matplotlib
import numpy
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn


matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    # binary classification of a digit as either 0 or 8
    minus, plus = 0, 8
    train_perm = numpy.where((labels[:60000] == minus) | (labels[:60000] == plus))[0]
    train_idx = numpy.random.RandomState(0).permutation(train_perm)
    test_perm = numpy.where((labels[60000:] == minus) | (labels[60000:] == plus))[0]
    test_idx = numpy.random.RandomState(0).permutation(test_perm) + 60000

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == (minus or plus)) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == (minus or plus)) * 2 - 1

    test_data_unscaled = data[test_idx, :].astype(float)
    test_labels = (labels[test_idx] == (minus or plus)) * 2 - 1

    # mean-center each coordinate of our data points
    train_data = sklearn.preprocessing.scale(train_data_unscaled, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, with_std=False)

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def sgd_svm_prediction(w, x):
    prediction = (numpy.inner(w, x) >= 0) * 2 - 1
    return prediction


def sgd_svm(x, y, w, C, eta):
    """implements a step of the sgd_svm algorithm, the weight vector w.
    w_t+1 = (1-eta)w_t + etaCyx """
    if y * numpy.inner(w, x) < 1:
        w = numpy.float64(((1-eta)*w)) + numpy.float64((eta*C*y*x))
    else:
        w = numpy.float64((1-eta)*w)
    return w


# noinspection PyPep8Naming
def sgd_svm_train(xs, ys, C, eta0, T):
    w = numpy.zeros(784)
    for t in xrange(1, T+1):
        i = numpy.random.randint(len(xs))
        x = xs[i]
        y = ys[i]
        eta = numpy.float64(eta0) / t
        w = sgd_svm(x, y, w, C, eta)
    return w


def accuracy_sgd_svm(xs, ys, w):
    predictions = numpy.array([sgd_svm_prediction(w, x) for x in xs])
    return numpy.mean(predictions == ys)


# noinspection PyPep8Naming
def part_abcd():
    numpy.random.seed(1)
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_data()
    runs = 10
    # noinspection PyPep8Naming
    T = 1000
    C = 1.0
    powers = numpy.arange(-5, 3, 1./5)
    eta0 = [10 ** float(power) for power in powers]
    accuracies = []

    for eta in eta0:
        accuracy = []
        for _ in xrange(runs):
            w = sgd_svm_train(train_data, train_labels, C, eta, T)
            accuracy.append(accuracy_sgd_svm(test_data, test_labels, w))
        accuracies.append(accuracy)

    # Plot the average accuracy on the validation set, as a function of eta0.
    plt.xlabel('log(eta0)')
    plt.ylabel('average accuracy')
    plt.title('SGD SVM')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(a)')

    plt.scatter(numpy.log10(eta0), numpy.mean(accuracies, axis=1), marker='o', label='validation accuracy')
    plt.legend(loc=3)
    plt.savefig('q1_part_a.png')
    plt.clf()

    max_eta_ind = numpy.argmax([numpy.mean(arr) for arr in accuracies])
    eta_best = eta0[max_eta_ind]
    print 'The best eta0 is achieved for: {}'.format(str(eta_best))

    powers = numpy.arange(-8, 5, 1. / 5)
    C0 = [10 ** float(power) for power in powers]
    accuracies = []

    for c in C0:
        accuracy = []
        for _ in xrange(runs):
            w = sgd_svm_train(train_data, train_labels, c, eta_best, T)
            accuracy.append(accuracy_sgd_svm(test_data, test_labels, w))
        accuracies.append(accuracy)

    # Plot the average accuracy on the validation set, as a function of C.
    plt.xlabel('log(C)')
    plt.ylabel('average accuracy')
    plt.title('SGD SVM')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(b)')

    plt.scatter(numpy.log10(C0), numpy.mean(accuracies, axis=1), marker='o', label='validation accuracy')
    plt.legend(loc=3)
    plt.savefig('q1_part_b.png')
    plt.clf()

    max_c_ind = numpy.argmax([numpy.mean(arr) for arr in accuracies])
    c_best = C0[max_c_ind]
    print 'The best C is achieved for: {}'.format(str(c_best))

    T = 20000
    w = sgd_svm_train(train_data, train_labels, c_best, eta_best, T)
    numpy.save('q1_part_c.png', w)
    plt.imshow(numpy.reshape(w, (28, 28)), interpolation='nearest')
    plt.title('w as a picture')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(c)')
    plt.savefig('q1_part_c.png')

    accuracy_best = accuracy_sgd_svm(test_data, test_labels, w)
    print 'The accuracy of the best classifier on the test set is: {}'.format(accuracy_best)


if __name__ == "__main__":
    help_text = """ Run question 3 all parts. No parameters are required. """
    parser = OptionParser(epilog=help_text)
    (_, _) = parser.parse_args()

    print 'Running Question 1 all parts...\n'
    part_abcd()
