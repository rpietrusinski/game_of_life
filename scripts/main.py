import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import argparse


# argparse
parser = argparse.ArgumentParser(description="Pass game parameters.")
parser.add_argument('-d', '--dim', type=int, help="Dimension of game")
parser.add_argument('-n', '--n_iter', type=int, help="Number of game iterations")
args = parser.parse_args()
dim = args.dim
n_iter = args.n_iter


def generate_start_population(dim_: int) -> np.array:
    """Generates start population

    :param dim_: Dimension
    :return: array with zeroes and ones indicating living and dead cells
    """
    x_t0 = np.random.choice([0, 1], size=(dim_ ** 2,), p=[.9, .1]).reshape(dim_, dim_)
    return x_t0


def life_step_1(x: np.array) -> np.array:
    """Runs single iteration. Uses generator with numpy array roll. Can be used interchangeably with life_step_2().

    :param x: t0 population array
    :return: t1 population array
    """
    neighbors = sum(np.roll(np.roll(x, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1) if i != 0 or j != 0)
    return (neighbors == 3) | (x & (neighbors == 2))


def life_step_2(x: np.array) -> np.array:
    """Runs single iteration. Uses 2d convolution. Can be used interchangeably with life_step_1().

    :param x: t0 population array
    :return: t1 population array
    """
    filter_matrix = np.ones((3, 3))
    filter_matrix[1, 1] = 0
    neighbors = convolve2d(x, filter_matrix, mode='same', boundary='wrap')
    return (neighbors == 3) | (x & (neighbors == 2))


def run_experiment(dim_: int, n_iter_: int) -> None:
    """Runs experiment

    :param dim_: Dimension
    :param n_iter_: Number of iterations
    :return: None
    """

    popul = generate_start_population(dim_)
    fig, ax = plt.subplots(1, figsize=(20, 20))
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    for i in range(n_iter_):
        ax.imshow(popul, cmap='Greys')
        ax.set_title("Iter: {}".format(i))
        plt.pause(.5)
        next_popul = life_step_2(popul)
        popul = next_popul
    plt.pause(10)
    plt.close()


if __name__ == '__main__':
    run_experiment(dim, n_iter)
