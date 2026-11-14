import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

np.random.seed(100)

class PlottingTool:

    def __init__(self, x0: np.ndarray, v0: np.ndarray, d=3, agents=None, structure="base"):
        self.d = d
        self.agents = agents or np.random.randint(20,40)
        self.x0, self.v0 = x0, v0
        self.structure = structure

        if self.d == 3:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection="3d")
        else:
            self.fig, self.ax = plt.subplots()

        self.functions = {
            "base": self.plot_edges_base,
            "line": self.plot_edges_line,
            "ring": self.plot_edges_ring,
            "star": self.plot_edges_star
        }

        self.A_init = np.zeros((self.agents, self.agents))

    def plot_edges_base(self):
        for i, j in combinations(range(self.agents), 2):
            self.A_init[i,j] = self.A_init[j,i] = 1
            self.ax.plot(*zip(self.x0[i], self.x0[j]), lw=0.5)
        return self.A_init

    def plot_edges_line(self):
        for i in range(self.agents - 1):
            self.A_init[i,i+1] = self.A_init[i+1,i] = 1
            self.ax.plot(*zip(self.x0[i], self.x0[i + 1]), lw=0.5)
        return self.A_init

    def plot_edges_ring(self):
        for i in range(self.agents):
            self.A_init[i, (i+1)%self.agents] = self.A_init[(i+1)%self.agents, i] = 1
            self.ax.plot(*zip(self.x0[i], self.x0[(i + 1) % self.agents]), lw=0.5)
        return self.A_init

    def plot_edges_star(self):
        center_index = self.agents // 2
        center = self.x0[center_index]
        for i in range(self.agents):
            if i != self.agents // 2:
                self.A_init[center_index, i] = self.A_init[i, center_index] = 1
                self.ax.plot(*zip(center, self.x0[i]), lw=0.5)
        return self.A_init

    def plot(self):
        A = self.functions[self.structure]()
        self.ax.scatter(*self.x0.T)
        return self.fig, self.ax, A


if __name__ == "__main__":
    x0, v0 = (
        np.random.normal(0, 1, size=(20,3)),
        np.random.normal(0, 1, size=(20,3))
    )
    fig, ax, A_init = PlottingTool(d=3, x0=x0, v0=v0, structure="star", agents=20).plot()
    plt.show()