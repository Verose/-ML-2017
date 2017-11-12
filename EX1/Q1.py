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
    for x in xs:
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
    plt.legend()
    plt.title('Sample points from probability')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(a)')

    plt.scatter(xs, ys)
    for x_tick in [0.25, 0.5, 0.75]:
        plt.plot([x_tick, x_tick], [-0.1, 1.1], 'r')
    for interval in intervals:
        plt.plot(interval, [0.5, 0.5], 'b')

    plt.show()


def calculate_true_error(intervals):
    # todo: calculate true error
    return 0.5


def part_c():
    T = 100
    m_range = range(10, 101, 5)
    average_empirical_error = []
    average_true_error = []
    for m in m_range:
        sum_empirical_error = 0.0
        sum_true_error = 0.0
        for _ in range(1, 101):  # T = 100
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
    plt.legend()
    plt.title('Question 1(a)')
    fig = plt.gcf()
    fig.canvas.set_window_title('Programming Assignment: Question 1(c)')

    plt.scatter(m_range, average_empirical_error, marker='o', label='empirical')
    plt.scatter(m_range, average_true_error, marker='+', label='true')
    plt.show()


def part_d():
    pass


def part_e():
    pass


if __name__ == "__main__":
    part_a()
