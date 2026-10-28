import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import distance
from typing import Callable

from models.cucker_smale.cucker_smale_setup import equations

# Default psi: will probably need to experiment how to change this / penalize extreme distances
def psi(r: np.ndarray) -> np.ndarray:
    return 1 / (1 + r**2) ** 0.25

class CuckerSmaleSolver:

    def __init__(self, x0: np.ndarray, v0: np.ndarray, psi: Callable = psi):
        self.x0 = x0
        self.v0 = v0
        self.psi = psi
        self.N, self.d = x0.shape
        self.solution = None

    # solve_ivp only takes a flattened matrix, so this takes positions and velocities and puts it into one flattened matrix
    def pack(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.concatenate([x.ravel(), v.ravel()])

    def unpack(self, y: np.ndarray):
        n = self.N * self.d
        return y[:n].reshape(self.N, self.d), y[n:].reshape(self.N, self.d)

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
        n = self.N * self.d
        return self.solution.y[:n].T.reshape(-1, self.N, self.d)

    @property
    def v(self) -> np.ndarray:
        n = self.N * self.d
        return self.solution.y[n:].T.reshape(-1, self.N, self.d)


if __name__ == "__main__":
    x0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1,1,1]], dtype=float)
    v0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [2,2,2]], dtype=float)

    solver = CuckerSmaleSolver(x0, v0).solve(10000, 100) 
    pairwise_distances = distance.squareform(distance.pdist(solver.v[-1], metric="euclidean"))
    print(pairwise_distances)