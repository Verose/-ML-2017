from optparse import OptionParser

import EX4.backprop.data
import EX4.backprop.network


def part_b():
    training_data, test_data = EX4.backprop.data.load(train_size=10000, test_size=5000)
    net = EX4.backprop.network.Network([784, 40, 10])
    net.SGD_part_b(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


def part_c():
    training_data, test_data = EX4.backprop.data.load(train_size=50000, test_size=10000)
    net = EX4.backprop.network.Network([784, 40, 10])
    net.SGD_part_c(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)


def part_d():
    training_data, test_data = EX4.backprop.data.load(train_size=10000, test_size=5000)
    net = EX4.backprop.network.Network([784, 30, 30, 30, 30, 10])
    net.SGD_part_d(training_data, epochs=30, mini_batch_size=10000, learning_rate=0.1, test_data=test_data)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--b", action='store_true', help="Run question 2 part (b)")
    parser.add_option("--c", action='store_true', help="Run question 2 part (c)")
    parser.add_option("--d", action='store_true', help="Run question 2 part (d)")
    (options, args) = parser.parse_args()

    if options.b:
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
