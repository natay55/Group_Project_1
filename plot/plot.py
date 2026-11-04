import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

class PlottingTool:

    def __init__(self, x0: np.ndarray, v0: np.ndarray, d=3, agents=None, structure = "base"):
        self.d = d
        self.agents = agents or np.random.randint(20, 40)
        self.x0, self.v0 = x0, v0
        self.structure = structure

        if self.d == 3:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection="3d")
        else:
            self.fig, self.ax = plt.subplots()

        # Used to switch between different graph structures without issue
        self.functions = {
            "base": self.plot_edges_base,
            "line": self.plot_edges_line, 
            "ring": self.plot_edges_ring,
            "star": self.plot_edges_star
        }
    
    # Base case for a weighted, undirected, complete graph (default structure)
    def plot_edges_base(self):
        for i, j in combinations(range(self.agents), 2):
            self.ax.plot(*zip(self.x0[i], self.x0[j]), lw=0.5)

    # Line graph - birds connected to, at most, two nodes
    def plot_edges_line(self):
        for i in range(self.agents - 1):
            self.ax.plot(*zip(self.x0[i], self.x0[i + 1]), lw=0.5)

    # Ring graphs - birds connected to, at most, two nodes, but the first and last nodes are also connected
    def plot_edges_ring(self):
        for i in range(self.agents):
            self.ax.plot(*zip(self.x0[i], self.x0[(i + 1) % self.agents]), lw=0.5)

    # Star graph - one bird connected to all other birds, but the other birds are not connected to each other
    def plot_edges_star(self):
        center = self.x0[self.agents // 2]
        for i in range(self.agents):
            if i != self.agents // 2:
                self.ax.plot(*zip(center, self.x0[i]), lw=0.5)

    def plot(self):
        self.functions[self.structure]()
        self.ax.scatter(*self.x0.T)
        return self.fig, self.ax

if __name__ == "__main__":
    PlottingTool(d=3, structure="star").plot()
    plt.show()