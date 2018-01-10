from optparse import OptionParser

import matplotlib
import numpy
import numpy.random
import sklearn
from sklearn.datasets import fetch_mldata

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

    train_data_unscaled = data[train_idx, :].astype(float)
    train_labels = (labels[train_idx] == (minus or plus)) * 2 - 1

    test_data_unscaled = data[test_idx, :].astype(float)
    test_labels = (labels[test_idx] == (minus or plus)) * 2 - 1

    # mean-center each coordinate of our data points
    train_data = sklearn.preprocessing.scale(train_data_unscaled, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, with_std=False)

    return train_data, train_labels, test_data, test_labels


# The principal components are the r eigenvectors of Sigma with the largest eigenvalues.
def PCA(data, k):
    """Perform PCA on a set of images, input the dimension k of the PCA, and output the set of
eigenvectors and their corresponding eigenvalues"""
    # first step - remove mean
    data -= numpy.mean(data)
    X = data
    sigma = numpy.dot(X.T, X)

    # map x in Rd -> a in Rk
    eigenvalues, eigenvectors = numpy.linalg.eigh(sigma)

    # sort in decreasing order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # return the k eigenvectors with the largest eigenvalues
    return eigenvalues[:k], eigenvectors[:, :k]


def part_abcdef():
    train_data, train_labels, test_data, test_labels = load_data()

    # part a+b
    for sign in [8, 0]:
        label = train_labels == 1 if sign == 8 else -1
        signed_labels = train_data[train_labels == label]
        mean_img = numpy.mean(signed_labels, axis=0)
        plt.imshow(numpy.reshape(mean_img, (28, 28)), interpolation='nearest')
        plt.savefig('q1_part_a_mean_image_{}.png'.format(sign))
        plt.clf()

        eigenvalues, eigenvectors = PCA(signed_labels, k=5)
        for i, eigvec in enumerate(eigenvectors.T):
            plt.imshow(eigvec.reshape(28, 28), interpolation='nearest')
            plt.savefig('q1_part_a_first_{}_5_eigenvectors_{}.png'.format(i+1, sign))
            plt.clf()

        eigenvalues, eigenvectors = PCA(signed_labels, k=100)

        # Plot the eigenvalues (in decreasing order) as a function of dimension (for the first 100 dimensions)
        plt.xlabel('dimension')
        plt.ylabel('eigenvalues')
        plt.title('Eigenvalues as a function of Dimension')
        fig = plt.gcf()
        fig.canvas.set_window_title('Programming Assignment: Question 1(a+b)')

        plt.scatter(xrange(100), eigenvalues, marker='o')
        plt.legend()
        plt.savefig('q1_part_a_eigenvalues_vs_dimension_{}.png'.format(sign))
        plt.clf()

    # part c
    mean_img = numpy.mean(train_data, axis=0)
    plt.imshow(numpy.reshape(mean_img, (28, 28)), interpolation='nearest')
    plt.savefig('q1_part_c_mean_image.png')
    plt.clf()

    eigenvalues, eigenvectors = PCA(train_data, k=5)
    for i, eigvec in enumerate(eigenvectors.T):
        plt.imshow(eigvec.reshape(28, 28), interpolation='nearest')
        plt.savefig('q1_part_c_first_{}_5_eigenvectors.png'.format(i + 1))
        plt.clf()

    eigenvalues, eigenvectors = PCA(train_data, k=100)

    # Plot the eigenvalues (in decreasing order) as a function of dimension (for the first 100 dimensions)
    plt.xlabel('dimension')
    plt.ylabel('eigenvalues')
    plt.title('Eigenvalues as a function of Dimension')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(c)')

    plt.scatter(xrange(100), eigenvalues, marker='o')
    plt.legend()
    plt.savefig('q1_part_c_eigenvalues_vs_dimension.png')
    plt.clf()

    # part d
    _, eigenvectors = PCA(train_data, k=2)
    proj = numpy.dot(train_data, eigenvectors)
    pos = proj[train_labels == 1]
    neg = proj[train_labels == -1]

    # 2d scatterplot showing the projections of the images on the first two principal axes
    plt.xlabel('proj on 1st principal axis')
    plt.ylabel('proj on 2nd principal axis')
    plt.title('Projections of the images on the first two principal axes')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(d)')

    plt.scatter(pos[:, 0], pos[:, 1], marker='.', c='red', label='positive (8)')
    plt.scatter(neg[:, 0], neg[:, 1], marker='.', c='blue', label='negative (0)')
    plt.legend()
    plt.savefig('q1_part_d_proj_two_axes.png')
    plt.clf()

    # part e
    _, eigenvectors = PCA(train_data, k=50)
    X = numpy.concatenate((train_data[train_labels == 1][:2], train_data[train_labels == -1][:2]))

    for i, image in enumerate(X):
        plt.imshow(image.reshape(28, 28), interpolation='nearest')
        plt.savefig('q1_part_e_original_{}.png'.format(i + 1))
        plt.clf()

    for k in [10, 30, 50]:
        V = eigenvectors[:, :k]
        reconstruction = numpy.dot(numpy.dot(V, V.T), X.T)

        for i, image in enumerate(reconstruction.T):
            plt.imshow(image.reshape(28, 28), interpolation='nearest')
            plt.savefig('q1_part_e_reconstructed_{}.png'.format(i + 1))
            plt.clf()

    # part f
    _, eigenvectors = PCA(train_data, k=len(train_data))
    dimensions = xrange(1, len(train_data[0]))
    losses = []

    for dim in dimensions:
        X = train_data[:dim]
        reconstruction = numpy.dot(numpy.dot(V, V.T), X.T)
        losses += [numpy.sum([numpy.linalg.norm(x - reco) for x, reco in zip(X, reconstruction.T)])]

    # Plot the PCA objective as a function of k
    plt.xlabel('dimension')
    plt.ylabel('objective')
    plt.title('PCA objective as a function of k')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(f)')

    plt.scatter(dimensions, losses, marker='.')
    plt.legend()
    plt.savefig('q1_part_f_pca_objective.png')
    plt.clf()


if __name__ == "__main__":
    help_text = """ Run question 1 all parts. No parameters are required. """
    parser = OptionParser(epilog=help_text)
    (_, _) = parser.parse_args()

    print 'Running Question 1 all parts...\n'
    part_abcdef()
