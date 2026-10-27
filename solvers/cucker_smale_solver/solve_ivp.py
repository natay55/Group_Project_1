import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.integrate import solve_ivp
from models.cucker_smale.cucker_smale_setup import equations

x0 = np.array(([1,0,0], [0,1,0], [0,0,1]))
v0 = np.array(([1,0,0], [0,1,0], [0,0,1]))
A_init = np.array(([0,1,0], [0,0,1], [0,0,0]))
N = x0.shape[0]
psi = lambda r: 1 / (1+r**2)**0.25 # Citation from paper needed (https://people.mpi-inf.mpg.de/~mehlhorn/SeminarEvolvability/CuckerSmale.pdf)

y0 = np.concatenate([x0.ravel(), v0.ravel()])

def ode(t, y, A_init=np.zeros(shape=(3,3))):

    x = y[:N*4].reshape(N, 4)
    v = y[N*4:].reshape(N, 4)

    dx, dv = equations(x, v, psi)

    return (np.concatenate([dx.ravel(), dv.ravel()]))

t = np.linspace(0,10000)

solve = solve_ivp(
    ode,
    (0, 10000),
    y0=y0,
    t_eval=t
)

x = solve.y[:N*4].T.reshape(-1, N , 4)
v = solve.y[N*4:].T.reshape(-1, N, 4)[-1]
#print(np.linalg.eigvals(v))
# print(distance.squareform(distance.pdist(v, metric="euclidean")))


