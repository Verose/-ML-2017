import matplotlib.pyplot as plt
import numpy
from intervals import find_best_interval


def sample_points_from_probability(size):
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
    xs, ys = sample_points_from_probability(sample_size)
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


def calculate_true_error(intervals):
    # denote: A: [0, 0.25] U [0.5, 0.75]
    # denote: B: [0.25, 0.5] U [0.75, 1]
    len_a = 0
    len_b = 0

    for interval in intervals:
        if 0 <= interval[0] <= 0.25:
            len_a += max(0.0, min(0.25, interval[1]) - max(0, interval[0]))
        if 0.25 < interval[0] < 0.75:
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
    average_empirical_error = []
    average_true_error = []
    for m in m_range:
        sum_empirical_error = 0.0
        sum_true_error = 0.0
        for _ in range(1, T + 1):
            #  (i) Draw a sample of size m and run the ERM algorithm on it
            xs, ys = sample_points_from_probability(m)
            intervals, best_error = find_best_interval(xs, ys, k=2)

            #  (ii) Calculate the empirical error for the returned hypothesis
            sum_empirical_error += float(best_error) / m

            #  (iii) Calculate the true error for the returned hypothesis
            sum_true_error += calculate_true_error(intervals)
        average_empirical_error += [sum_empirical_error / T]
        average_true_error += [sum_true_error / T]

    # Plot the average empirical and true errors, averaged across the T runs, as a function of m
    plt.xlabel('m')
    plt.ylabel('error')
    plt.title('Empirical error vs True error')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(c)')

    plt.scatter(m_range, average_empirical_error, marker='o', label='empirical error')
    plt.scatter(m_range, average_true_error, marker='+', label='true error')
    plt.legend()
    plt.savefig('q1_part_c.png')
    plt.clf()


def part_d():
    sample_size = 50
    xs, ys = sample_points_from_probability(sample_size)
    results = []
    for i in range(1, 21):
        intervals, best_error = find_best_interval(xs, ys, k=20)
        true_error = calculate_true_error(intervals)
        results.append((best_error, true_error))
    pass


def part_e():
    pass


if __name__ == "__main__":
    part_c()
