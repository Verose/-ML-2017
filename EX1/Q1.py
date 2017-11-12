import matplotlib.pyplot as plt
import numpy
from intervals import find_best_interval


def a_plot_points_from_probability(
        # P[y = 1|x] = (
        # 0.8 if x in [0, 0.25] or x in [0.5, 0.75]
        # 0.1 if x in [0.25, 0.5] or x in [0.75, 1]
):
    xs = numpy.random.uniform(size=100)
    ys = numpy.array([])
    xs.sort()
    for x in xs:
        if 0 <= x <= 0.25 or 0.5 <= x <= 0.75:
            y = numpy.random.choice(numpy.arange(0, 2), p=[0.2, 0.8])
        else:
            y = numpy.random.choice(numpy.arange(0, 2), p=[0.9, 0.1])
        ys = numpy.append(ys, y)
    plt.xticks([0, 0.25, 0.5, 0.75, 1])

    plt.scatter(xs, [y - 0.1 if y == 0 else y + 0.1 if y == 1 else y for y in ys])  # shift y on edges
    intervals, best_error = find_best_interval(xs, ys, 2)
    plt.plot(intervals, [(0, 0), (0, 0)], marker='o')  # todo: print interval correctly

    plt.show()
    print 'Great success'


def b():
    for m in range(10, 101, 5):
        k =2
        for i in range(1, 101):  # T = 100
            #  (i) Draw a sample of size m and run the ERM algorithm on it
            xs = numpy.random.uniform(size=m)
            ys = numpy.array([])
            xs.sort()
            for x in xs:
                if 0 <= x <= 0.25 or 0.5 <= x <= 0.75:
                    y = numpy.random.choice(numpy.arange(0, 2), p=[0.2, 0.8])
                else:
                    y = numpy.random.choice(numpy.arange(0, 2), p=[0.9, 0.1])
                ys = numpy.append(ys, y)
            intervals, best_error = find_best_interval(xs, ys, 2)

            #  (ii) Calculate the empirical error for the returned hypothesis
            empirical_error = best_error / m

            #  (iii) Calculate the true error for the returned hypothesis


# Plot the average empirical and true errors, averaged across the T runs, as a function of m


if __name__ == "__main__":
    a_plot_points_from_probability()
