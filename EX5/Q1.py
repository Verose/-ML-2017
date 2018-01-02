from optparse import OptionParser

import matplotlib

from hw5 import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# noinspection PyPep8Naming
def erm_for_decision_stumps(xy_dist):
    F_opt = float('inf')
    theta_opt = 0
    j_opt = 0

    d = len(xy_dist[0][0])
    for j in range(d):
        sorted_xy_dist = sorted(xy_dist, key=lambda x: x[0][j])
        m = len(sorted_xy_dist) - 1
        xs = [xyd[0] for xyd in sorted_xy_dist]
        ys = [xyd[1] for xyd in sorted_xy_dist]
        dists = [xyd[2] for xyd in sorted_xy_dist]

        last_item = sorted_xy_dist[-1]
        last_item[0][j] = last_item[0][j] + 1
        sorted_xy_dist = sorted_xy_dist + [last_item]
        F = sum([dists[index] if ys[index] == 1 else 0 for index in range(m)])

        if F < F_opt:
            F_opt = F
            theta_opt = sorted_xy_dist[0][0][j] - 1
            j_opt = j

        for i in range(m):
            F = F - ys[i] * dists[i]
            if F < F_opt and xs[i][j] != xs[i + 1][j]:
                F_opt = F
                theta_opt = 0.5 * (xs[i][j] + xs[i + 1][j])
                j_opt = j

    return j_opt, theta_opt


# noinspection PyPep8Naming
def adaboost_weak_learner(w, h, xs, T):
    return sign(sum([w[i] * (1 if xs[h[i][0]] <= h[i][1] else -1) for i in range(T)]))


# noinspection PyPep8Naming
def part_ab():
    T = 100
    T_range = xrange(T)
    distribution = ones(train_data_size, dtype=float64) / train_data_size
    xy_dist = [[x, y, dist] for x, y, dist in zip(train_data, train_labels, distribution)]
    j_theta = [[0, 0] for _ in range(T)]
    weights = zeros(T)
    train_accuracy = []
    test_accuracy = []

    for it in T_range:
        j_theta[it][0], j_theta[it][1] = erm_for_decision_stumps(xy_dist)
        predictions = [1 if x[j_theta[it][0]] <= j_theta[it][1] else -1 for x, _, _ in xy_dist]
        epsilon = sum([dist if predictions[i] != y else 0 for i, (_, y, dist) in enumerate(xy_dist)])
        weights[it] = 0.5 * log((1-epsilon)/epsilon) if epsilon != 0 else 0
        dist_sum = 0

        for i in xrange(len(xy_dist)):
            xy_dist[i][2] = xy_dist[i][2] * exp((-1) * weights[it] * xy_dist[i][1] * predictions[i])
            dist_sum += xy_dist[i][2]
        xy_dist = [[x, y, dist / dist_sum] for x, y, dist in xy_dist]

        train_pred = [adaboost_weak_learner(weights, j_theta, x, it + 1) for x, y in zip(train_data, train_labels)]
        test_pred = [adaboost_weak_learner(weights, j_theta, x, it + 1) for x, y in zip(test_data, test_labels)]

        train_accuracy += [mean(train_pred == train_labels)]
        test_accuracy += [mean(test_pred == test_labels)]

    # Plot (on one graph) the training error and the test error of the classifier
    plt.xlabel('t')
    plt.ylabel('error')
    plt.title('Train & Test accuracy per iteration')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1')

    plt.plot(T_range, train_accuracy, label='train accuracy')
    plt.plot(T_range, test_accuracy, label='test accuracy')
    plt.legend()
    plt.savefig('q1_accuracy.png')
    plt.clf()


if __name__ == '__main__':
    help_text = """ Run question 1 all parts. No parameters are required. """
    parser = OptionParser(epilog=help_text)
    (_, _) = parser.parse_args()

    print 'Running Question 1 all parts...\n'
    part_ab()
