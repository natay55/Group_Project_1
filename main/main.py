import numpy as np 
import matplotlib.pyplot as plt
from solvers.cucker_smale_solver.solve_ivp import CuckerSmaleSolver
from scipy.spatial import distance
from plot.plot import PlottingTool

generate_coordinates = lambda: ((np.random.normal(0,1,(20,3)), np.random.normal(0,1,(20,3))))
x0, v0 = generate_coordinates()
structure = "ring"
fig, ax, A_init = PlottingTool(x0, v0, agents=20, structure=structure).plot()

if __name__ == "__main__":

    solve_system = CuckerSmaleSolver(x0=x0, v0=v0, A_init=A_init).solve(t_end=100, n_points=100)
    pairwise_distances = np.array([
        distance.squareform(distance.pdist(v, metric="euclidean"))
        for v in solve_system.v
    ])

    # Plotting the worst case convergence rate using maximum pairwise velocity and mean pairwise velocity
    max_distances = np.max(pairwise_distances, axis=(1,2))
    mean_distances = np.sum(pairwise_distances, axis=(1,2)) / (solve_system.agents * (solve_system.agents - 1))

    fig, ax = plt.subplots()
    plt.plot(solve_system.solution.t, max_distances, label="Maximum Norm at timestep 't'")
    plt.plot(solve_system.solution.t, mean_distances, label="Mean Norm")
    plt.xlabel("Time")
    plt.ylabel("Pairwise Velocity Distance")
    plt.title(f'Pairwise Velocity Norm Over Time For {structure.capitalize()} Network')
    plt.show()
    







    
    


