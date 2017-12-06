import matplotlib
import numpy
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

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


def part_abcde():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = load_data()

    print 'part a: SVM cross-validation accuracy on the training & validation sets:'
    powers = numpy.arange(-10, 10)
    # noinspection PyPep8Naming
    C = [10 ** int(power) for power in powers]
    train_scores = []
    validation_scores = []

    for c in C:
        clf = LinearSVC(loss='hinge', C=c, fit_intercept=False, random_state=0)
        clf.fit(train_data, train_labels)
        train_scores += [clf.score(train_data, train_labels)]
        validation_scores += [clf.score(validation_data, validation_labels)]

    # Plot the accuracy of the resulting hyperplane on the training set and on the validation set, as a function of C
    plt.xlabel('log(C)')
    plt.ylabel('accuracy')
    plt.title('Prediction accuracy as a function of C')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 3(a)')

    plt.plot(numpy.log10(C), train_scores, marker='o', label='Train Accuracy')
    plt.plot(numpy.log10(C), validation_scores, marker='+', label='Validation Accuracy')
    plt.legend()
    plt.savefig('q3_part_a.png')
    plt.clf()

    best_score = numpy.argmax(validation_scores)
    best_c = C[best_score]
    print 'The best C is: 10^{} with accuracy {}'.format(numpy.log10(best_c), validation_scores[best_score])

    print '\npart c: w for the best C as an image:'
    clf = LinearSVC(loss='hinge', C=best_c, fit_intercept=False, random_state=0)
    clf.fit(train_data, train_labels)
    w = clf.coef_
    plt.imshow(numpy.reshape(w, (28, 28)), interpolation='nearest')
    plt.title('w as a picture')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 3(c)')
    plt.savefig('q3_part_c.png')

    print '\npart d: accuracy of the linear SVM with the best C on the test set:'
    test_score = clf.score(test_data, test_labels)
    print 'The accuracy with the best C on the test set is: {}'.format(test_score)

    print '\npart e: RBF kernel SVM, training set & test set accuracy:'
    clf = SVC(C=10, gamma=5*(10**-7), random_state=0)
    clf.fit(train_data, train_labels)
    rbf_train_score = clf.score(train_data, train_labels)
    rbf_test_score = clf.score(test_data, test_labels)
    print 'The accuracy of RBF kernel SVM on the training set: {}, on the test set: {}'\
        .format(rbf_train_score, rbf_test_score)

    # summary
    print '\nsummary: '
    columns = ['train', 'test']
    df = pd.DataFrame([['', test_score], [rbf_train_score, rbf_test_score]],
                      ['SVM with best C on test set', 'SVM Accuracy with RBF Kernel'], columns)
    df.index.name = 'Accuracy rate'
    print df


if __name__ == "__main__":
    print 'Running Question 3 all parts...\n'
    part_abcde()
