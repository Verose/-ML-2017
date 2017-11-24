from optparse import OptionParser

import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from intervals import find_best_interval

plt.ioff()  # shut off interactive mode


def sample_points_from_distribution(size):
    """ P[y = 1|x] = (
        0.8 if x in [0, 0.25] or x in [0.5, 0.75]
        0.1 if x in [0.25, 0.5] or x in [0.75, 1]"""
    xs = numpy.random.uniform(size=size)
    ys = numpy.array([])
    xs.sort()
    for x in numpy.nditer(xs):
        if 0 <= x <= 0.25 or 0.5 <= x <= 0.75:
            y = numpy.random.choice(numpy.arange(0, 2), p=[0.2, 0.8])
        else:
            y = numpy.random.choice(numpy.arange(0, 2), p=[0.9, 0.1])
        ys = numpy.append(ys, y)
    return xs, ys


def part_a():
    sample_size = 100
    xs, ys = sample_points_from_distribution(sample_size)
    intervals, best_error = find_best_interval(xs, ys, k=2)

    plt.xticks([0, 0.25, 0.5, 0.75, 1])
    plt.yticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1, 1.1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Sample points from distribution')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(a)')

    plt.scatter(xs, ys, label='data points')
    for x_tick in [0.25, 0.5, 0.75]:
        plt.plot([x_tick, x_tick], [-0.1, 1.1], 'r')
    for i, interval in enumerate(intervals):
        if i == 0:
            plt.plot(interval, [0.5, 0.5], 'b', label='model')
        else:
            plt.plot(interval, [0.5, 0.5], 'b')

    plt.legend()
    plt.savefig('q1_part_a.png')
    plt.clf()


def calculate_empirical_error(intervals, xs, ys):
    empirical_error = 0
    for x, y in zip(xs, ys):
        point_in_interval = False
        for interval in intervals:
            if interval[0] < x < interval[1]:
                point_in_interval = True
        empirical_error += (y != point_in_interval)

    return empirical_error


def calculate_true_error(intervals):
    # denote: A: [0, 0.25] U [0.5, 0.75]
    # denote: B: [0.25, 0.5] U [0.75, 1]
    len_a = 0
    len_b = 0

    for interval in intervals:
        if 0 <= interval[0] <= 0.25:
            len_a += max(0.0, min(0.25, interval[1]) - max(0, interval[0]))
        if 0.25 <= interval[0] <= 0.75:
            len_a += max(0.0, min(0.75, interval[1]) - max(0.5, interval[0]))
            if 0.25 < interval[0] < 0.5:
                len_b += max(0.0, min(0.5, interval[1]) - max(0.25, interval[0]))
        if 0.25 < interval[1] < 0.5:
            len_b += max(0.0, interval[1] - max(0.25, interval[0]))
        if 0.75 < interval[1] < 1:
            len_b += max(0.0, interval[1] - max(0.75, interval[0]))
    # the error on A is 0.2 when predicting 1 and 0.8 when predicting 0 (on the remainder)
    # the error on B is 0.9 when predicting 1 and 0.1 when predicting 0 (on the remainder)
    total_error = len_a * 0.2 + len_b * 0.9 + (0.5 - len_a) * 0.8 + (0.5 - len_b) * 0.1

    return total_error


def part_c():
    T = 100
    m_range = range(10, 101, 5)
    empirical_error_average = []
    true_error_average = []
    for m in m_range:
        sum_empirical_error = 0.0
        sum_true_error = 0.0
        for _ in range(1, T + 1):
            #  (i) Draw a sample of size m and run the ERM algorithm on it
            xs, ys = sample_points_from_distribution(m)
            intervals, best_error = find_best_interval(xs, ys, k=2)

            #  (ii) Calculate the empirical error for the returned hypothesis
            sum_empirical_error += float(best_error) / m

            #  (iii) Calculate the true error for the returned hypothesis
            sum_true_error += calculate_true_error(intervals)
        empirical_error_average += [sum_empirical_error / T]
        true_error_average += [sum_true_error / T]

    # Plot the average empirical and true errors, averaged across the T runs, as a function of m
    plt.xlabel('m')
    plt.ylabel('error')
    plt.title('Empirical error vs True error')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(c)')

    plt.scatter(m_range, empirical_error_average, marker='o', label='empirical error')
    plt.scatter(m_range, true_error_average, marker='+', label='true error')
    plt.legend()
    plt.savefig('q1_part_c.png')
    plt.clf()


def part_d():
    # Find the best ERM hypothesis for k=1,2,...,20
    m = 50
    empirical_error = []
    true_error = []
    k_range = range(1, 21)
    xs, ys = sample_points_from_distribution(m)
    for k in k_range:
        intervals, best_error = find_best_interval(xs, ys, k=k)
        empirical_error += [float(best_error) / m]
        true_error += [calculate_true_error(intervals)]

    # plot the empirical and true errors as a function of k.
    plt.xlabel('k')
    plt.ylabel('errors')
    plt.title('Empirical error vs True error')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(d)')

    plt.scatter(k_range, empirical_error, marker='o', label='empirical error')
    plt.scatter(k_range, true_error, marker='+', label='true error')
    plt.legend()
    plt.savefig('q1_part_d.png')
    plt.clf()


def part_e_cross_validation():
    m = 50
    k_range = range(1, 21)
    chunks = 8
    empirical_error_average = []
    true_error_average = []
    xs, ys = sample_points_from_distribution(m)

    # Find the best ERM hypothesis for k=1,2,...,20
    for k in k_range:
        empirical_error = 0
        true_error = 0
        for chunk in range(chunks):
            chunk_start = chunk * m / chunks
            chunk_end = (chunk + 1) * m / chunks
            test_indices = numpy.arange(chunk_start, chunk_end)
            train_indices = numpy.array(range(0, chunk_start) + range(chunk_end, m))

            xs_train = xs[train_indices]
            ys_train = ys[train_indices]

            xs_test = xs[test_indices]
            ys_test = ys[test_indices]

            intervals, best_error = find_best_interval(xs_train, ys_train, k=k)
            true_error += calculate_true_error(intervals)
            empirical_error += calculate_empirical_error(intervals, xs_test, ys_test)
        empirical_error_average += [empirical_error * 1.0 / m]
        true_error_average += [true_error * 1.0 / m]
    # plot the empirical and true errors as a function of k.
    plt.xlabel('k')
    plt.ylabel('errors')
    plt.title('Empirical error vs True error')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(e)')

    plt.scatter(k_range, empirical_error_average, marker='o', label='empirical error')
    plt.scatter(k_range, true_error_average, marker='+', label='true error')
    plt.legend()
    plt.savefig('q1_part_e_cv.png')
    plt.clf()


def part_e_holdout():
    m = 50
    k_range = range(1, 21)
    empirical_error = []
    empirical_error_test = []
    true_error = []
    xs_test, ys_test = sample_points_from_distribution(m)
    xs_train, ys_train = sample_points_from_distribution(m)

    # Find the best ERM hypothesis for k=1,2,...,20
    for k in k_range:
        intervals, best_error = find_best_interval(xs_train, ys_train, k=k)
        curr_empirical_err = float(calculate_empirical_error(intervals, xs_test, ys_test))/(m)
        empirical_error += [curr_empirical_err]
        true_error += [float(calculate_true_error(intervals))]
        empirical_error_test += [best_error]
        print(curr_empirical_err)

    # plot the empirical and true errors as a function of k.
    plt.xlabel('k')
    plt.ylabel('errors')
    plt.title('Empirical error vs True error')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(e)')

    plt.scatter(k_range, empirical_error, marker='o', label='empirical error')
    plt.scatter(k_range, true_error, marker='+', label='true error')
    plt.legend()
    plt.savefig('q1_part_e.png')
    plt.clf()

    # plot the empirical and true errors as a function of k.
    plt.xlabel('k')
    plt.ylabel('errors')
    plt.title('Empirical error vs True error')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(e)')

    plt.scatter(k_range, empirical_error_test, marker='o', label='empirical_error_test error')
    plt.scatter(k_range, true_error, marker='+', label='true error')
    plt.legend()
    plt.savefig('q1_part_d.png')
    plt.clf()


def part_e():
    # Find the best ERM hypothesis for k=1,2,...,20
    m = 50
    empirical_error = []
    true_error = []
    minimum_empirical_error = 2*m
    best_hypothesis_empirical = -1
    best_hypothesis_true_error = -1
    minimum_true_error = 2.0
    best_hypothesis = -1
    k_range = range(1, 21)
    k_holdout_range = range(0, 50)
    xs, ys = sample_points_from_distribution(m)
    xs_holdout, ys_holdout = sample_points_from_distribution(m)
    for k in k_range:
        intervals, best_error = find_best_interval(xs, ys, k=k)
        curr_true_error = [calculate_true_error(intervals)]
        true_error += curr_true_error
        for k_holdout in k_holdout_range:
            holdout_empirical_error = calculate_empirical_error(intervals, xs_holdout, ys_holdout)
            if minimum_empirical_error > (holdout_empirical_error + best_error):
                minimum_empirical_error = holdout_empirical_error + best_error
                best_hypothesis_empirical = k
                best_hypothesis_true_error = curr_true_error

        print("true error: ", curr_true_error)
        print("minimum_true_error: ", minimum_true_error)
        if minimum_true_error > curr_true_error:
            minimum_true_error = curr_true_error
            best_hypothesis = k
        empirical_error += [float(best_error) / m]

    # plot the empirical and true errors as a function of k.
    plt.xlabel('k')
    plt.ylabel('errors')
    plt.title('Empirical error vs True error')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(e)')

    plt.plot(k_range, empirical_error, marker='o', label='empirical error')
    plt.plot(k_range, true_error, marker='+', label='true error')
    plt.legend()
    plt.savefig('q1_part_e.png')
    plt.clf()

    print("the minimum empirical error is: ", minimum_empirical_error)
    print("the minimum empirical error is in k: ", best_hypothesis_empirical)
    print("the minimum empirical error is with true error: ", best_hypothesis_true_error)
    print("the minimum true error is: ", minimum_true_error)
    print("best hypothesis is: ", best_hypothesis)

    pass


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--a", action='store_true', help="Run question 1 part (a)")
    parser.add_option("--c", action='store_true', help="Run question 1 part (c)")
    parser.add_option("--d", action='store_true', help="Run question 1 part (d)")
    parser.add_option("--e", action='store_true', help="Run question 1 part (e) using holdout validation")
    parser.add_option("--ecv", action='store_true', help="Run question 1 part (e) using cross validation")
    (options, args) = parser.parse_args()

    if options.a:
        print 'Running Question 1 part (a)...'
        part_a()
    elif options.c:
        print 'Running Question 1 part (c)...'
        part_a()
    elif options.d:
        print 'Running Question 1 part (d)...'
        part_d()
    elif options.e:
        print 'Running Question 1 part (e) with holdout validation...'
        part_e_holdout()
    elif options.ecv:
        print 'Running Question 1 part (e) with cross validation...'
        part_e_cross_validation()
