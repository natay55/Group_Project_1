import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import distance
from typing import Callable
from plot.plot import PlottingTool

from models.cucker_smale.cucker_smale_setup import equations

# Default psi: will probably need to experiment how to change this / penalize extreme distances
def psi(r: np.ndarray) -> np.ndarray:
    return 1 / (1 + r**2) ** 0.25

class CuckerSmaleSolver:

    def __init__(self, low=0, high=100, psi: Callable = psi, d = 3, agents = None):
        self.agents = agents or np.random.randint(20,40)
        self.low, self.high, self.d = low, high, d
        self.x0, self.v0 = self.generate_coordinates()
        self.psi = psi
        self.solution = None

    # Generate random starting co-ordinates determined from a uniform distribution in the domain (0, 100)
    def generate_coordinates(self):
        shape = (self.agents, self.d)
        return (np.random.uniform(self.low, self.high, shape),
                np.random.uniform(self.low, self.high, shape))

    # solve_ivp only takes a flattened matrix, so this takes positions and velocities and puts it into one flattened matrix
    def pack(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.concatenate([x.ravel(), v.ravel()])

    def unpack(self, y: np.ndarray):
        n = self.agents * self.d
        return y[:n].reshape(self.agents, self.d), y[n:].reshape(self.agents, self.d)

    def ode(self, t: float, y: np.ndarray) -> np.ndarray:
        x, v = self.unpack(y)
        dx, dv = equations(x, v, self.psi)
        return self.pack(dx, dv)

    def solve(self, t_end: float, n_points: int):
        t_eval = np.linspace(0, t_end, n_points)
        self.solution = solve_ivp(
            self.ode,
            (0, t_end),
            y0=self.pack(self.x0, self.v0),
            t_eval=t_eval,
        )
        return self

    @property
    def x(self) -> np.ndarray:
        n = self.agents * self.d
        return self.solution.y[:n].T.reshape(-1, self.agents, self.d)

    @property
    def v(self) -> np.ndarray:
        n = self.agents * self.d
        return self.solution.y[n:].T.reshape(-1, self.agents, self.d)


if __name__ == "__main__":

    solver = CuckerSmaleSolver().solve(10000, 100)
    pairwise_distances = distance.squareform(distance.pdist(solver.v[-1], metric="euclidean"))
    print(pairwise_distances)
