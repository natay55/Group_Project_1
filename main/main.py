
import numpy as np
import matplotlib.pyplot as plt
from solvers.cucker_smale_solver.solve_ivp import CuckerSmaleSolver
from scipy.spatial import distance
from plot.plot import PlottingTool

generate_coordinates = lambda: ((np.random.normal(0,1, size=(20,3)), np.random.normal(0,1,size=(20,3))))
TOL = 1e-3

if __name__ == "__main__":

    graphs = ["line"]
    results = {}

    for graph in graphs:
    
        x0, v0 = generate_coordinates()

        network_fig, network_ax, A_init = PlottingTool(x0, v0, agents=20, structure=graph).plot()
        max_norms = np.zeros((250, 100000))
        mean_norms = np.zeros((250, 100000))

        for generation in range(250):

            x0, v0 = generate_coordinates()

            solve_system = CuckerSmaleSolver(x0=x0,v0=v0,A_init=A_init).solve(t_end=100000, n_points=100000)
            pairwise_distances = np.array([distance.squareform(distance.pdist(v, metric="euclidean")) for v in solve_system.v])

            max_norms[generation, :] = np.max(pairwise_distances, axis=(1, 2))
            mean_norms[generation, :] = np.sum(pairwise_distances, axis=(1, 2)) / (20 * 19)

        overall_max = np.max(max_norms, axis=0)
        overall_mean = np.mean(mean_norms, axis=0)
        time_converge_max = np.where(overall_max < TOL)[0]
        time_converge_mean = np.where(overall_mean < TOL)[0]

        idx_max = time_converge_max[0] if len(time_converge_max) else None
        idx_mean = time_converge_mean[0] if len(time_converge_mean) else None

        t_max = solve_system.solution.t[idx_max] if idx_max is not None else None
        t_mean = solve_system.solution.t[idx_mean] if idx_mean is not None else None

        results[graph] = {
            "max": overall_max,
            "mean": overall_mean,
            "time": solve_system.solution.t
        }

        fig, ax = plt.subplots()

        ax.plot(
            results[graph]["time"],
            results[graph]["max"],
            label="Maximum across 250 simulations"
        )

        ax.plot(
            results[graph]["time"],
            results[graph]["mean"],
            label="Mean across 250 simulations"
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Pairwise velocity distance")
        ax.set_title(
            f"Pairwise Velocity Norm Over Time - {graph.capitalize()} Network"
        )

        if t_max is not None:
            ax.axvline(
                x=t_max,
                color = "blue",
                linestyle = "--",
                label = f"Max Consensus at Time {t_max:.0f}"
            )
        if t_mean is not None:
            ax.axvline(
                x=t_mean,
                color="orange",
                linestyle = "--",
                label = f"Mean Consensus at Time {t_mean:.0f}"
            )

        ax.legend()
        ax.grid(True)

    plt.show()
