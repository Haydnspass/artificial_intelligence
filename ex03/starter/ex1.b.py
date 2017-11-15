import numpy as np
import random
import matplotlib.pyplot as plt


def f(x):
    """
    Energy function.
    """
    return 0.2 * np.sin(12.5 * x) + (x - 1)**2 - 5


def minimize(x, y, start_position):
    def environment(x_i, boundaries, eps=2.5):
        while True:
            x_i = x_i + 2 * eps * np.random.random_sample() - eps
            if (x_i > boundaries[0]) and (x_i < boundaries[1]):
                break
        return x_i

    def free_energy(x1, x2, T):
        f_e = np.exp(-(f(x2) - f(x1))/T)
        return f_e
    """
    Minimize the energy function.
    :param x: array, x coordinates
    :param y: array, values of the energy function
    :param start_position: int, initial position of the agent
    :return: position with the minimal found value of energy function
    """
    best_pos = 0
    temp = 500
    alpha = 0.95

    #env = np.arange(x.__len__())
    for iter_num in range(500):
        temp = temp * alpha
        # move to different index of x by chance, so s is index not position
        while True:
            s = best_pos + np.random.choice([-2,-1,1,2])
            if s >= 0 and s <= x.__len__():
                break

        if f(x[s]) <= f(x[best_pos]):
            best_pos = s
        else:
            if free_energy(x[best_pos], x[s], temp) >= np.random.random_sample():
                best_pos = s

    assert 0 <= best_pos < len(y), 'incorrect index'
    return best_pos


def main():
    random.seed(2017)
    np.random.seed(2017)
    x = np.linspace(-0.5, 2, num=31, endpoint=True)
    y = f(x)
    print 'x = %s' % x
    print 'y = %s' % y
    start_position = 0

    best_pos = minimize(x, y, start_position)
    print "Best value %s at pos %d" % (y[best_pos], best_pos)

    if best_pos != np.argmin(y):
        print 'You haven\'t found the global minimum. Try harder!'
    else:
        print 'Success!'

    plt.plot(x, y)
    plt.plot(x, y, '--')
    plt.plot(x[start_position], y[start_position], '-bo', label='start pos', markersize=13)
    plt.plot(x[best_pos], y[best_pos], '-go', label='best found pos', markersize=13)
    plt.title('f(x)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
