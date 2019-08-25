import numpy as np
import numpy.random.common
import numpy.random.bounded_integers
import numpy.random.entropy
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
import tkinter as tk


class InputApp(tk.Tk):
    """User input interface.

    """
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("Game of Life")
        self.geometry("250x100")
        self.L1 = tk.Label(self, text="dim [int]")
        self.L1.config(font=("Arial", 14))
        self.E1 = tk.Entry(self)
        self.L2 = tk.Label(self, text="n_iter [int]")
        self.L2.config(font=("Arial", 14))
        self.E2 = tk.Entry(self)
        self.quit = tk.Button(self, text="OK", fg="red", command=self.on_button)
        self.quit.config(height=1, width=15)
        self.dim = None
        self.n_iter = None

        self.L1.grid(row=1, column=0)
        self.L2.grid(row=2, column=0)
        self.E1.grid(row=1, column=1)
        self.E2.grid(row=2, column=1)
        self.quit.grid(row=3, column=1)

    def on_button(self):
        self.dim = self.E1.get()
        self.n_iter = self.E2.get()
        self.destroy()


def generate_start_population(dim_: int) -> np.array:
    """Generates start population

    :param dim_: Dimension
    :return: array with zeroes and ones indicating living and dead cells
    """
    x_t0 = np.random.choice([0, 1], size=(dim_ ** 2,), p=[.5, .5]).reshape(dim_, dim_)
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
        plt.pause(.25)
        next_popul = life_step_2(popul)
        popul = next_popul
    plt.pause(5)
    plt.close()


if __name__ == '__main__':
    w = InputApp()
    w.mainloop()
    dim = int(w.dim)
    n_iter = int(w.n_iter)
    run_experiment(dim, n_iter)
