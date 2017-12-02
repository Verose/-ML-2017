from optparse import OptionParser

import matplotlib
import numpy
import numpy.random
from sklearn.datasets import fetch_mldata
from scipy.spatial import distance

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_data():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    return train, train_labels, test, test_labels


def KNN(images, labels, query_image, k):
    """implements the k-NN algorithm, returns a prediction of the query image, given the given label set of images"""
    # The KNN algorithm assigns a label based on the majority of the labels of the nearest neighbors to x.
    distances = []

    # find the euclidean distances (in the L2 norm) between the image in query and all other points
    for i in xrange(0, len(images)):
        distances += [distance.euclidean(images[i], query_image)]

    # sort according to distances and only take k nearest neighbours
    sorted_labels = [x for _, x in sorted(zip(distances, labels))]
    sorted_labels = sorted_labels[:k]

    counts = numpy.bincount(sorted_labels)
    prediction = numpy.argmax(counts)

    return prediction


def accuracy_knn(k, n):
    train_data, train_labels, test_data, test_labels = load_data()
    # Run the algorithm
    success = 0
    for test_image, test_label in zip(test_data, test_labels):
        prediction = KNN(train_data[:n], train_labels[:n], test_image, k)

        if prediction == test_label:
            success += 1

    # Accuracy of the prediction
    success_rate = (float(success) / len(test_data)) * 100
    return success_rate


def part_b():
    n = 1000
    k = 10

    success_rate = accuracy_knn(k, n)

    print 'The accuracy of KNN is {:.5g}%'.format(success_rate)


def part_c():
    n = 1000
    k_limit = 100
    accuracies = []

    acc_range = xrange(1, k_limit+1)
    for k in acc_range:
        accuracies += [accuracy_knn(k, n)]

    # Plot the prediction accuracy as a function of k
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('Prediction accuracy as a function of k')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(c)')

    plt.plot(acc_range, accuracies, label='Prediction Accuracy')
    plt.legend()
    plt.savefig('q1_part_c.png')
    plt.clf()


def part_d():
    k = 1
    n_limit = 5000
    accuracies = []

    acc_range = xrange(100, n_limit + 1, 100)
    for n in acc_range:
        accuracies += [accuracy_knn(k, n)]

    # Plot the prediction accuracy as a function of n
    plt.xlabel('n')
    plt.ylabel('accuracy')
    plt.title('Prediction accuracy as a function of n')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(d)')

    plt.plot(acc_range, accuracies, label='Prediction Accuracy')
    plt.legend()
    plt.savefig('q1_part_d.png')
    plt.clf()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--b", action='store_true', help="Run question 1 part (b)")
    parser.add_option("--c", action='store_true', help="Run question 1 part (c)")
    parser.add_option("--d", action='store_true', help="Run question 1 part (d)")
    (options, args) = parser.parse_args()

    if options.b:
        print 'Running Question 1 part (b)...'
        part_b()
    elif options.c:
        print 'Running Question 1 part (c)...'
        part_c()
    elif options.d:
        print 'Running Question 1 part (d)...'
        part_d()
    else:
        print 'Error: please run \'--help\' to see options'
