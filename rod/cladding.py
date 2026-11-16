import numpy as np


def compute_wall_heat_flux(q_vol, R_pellet, R_clad):
    q_wall = (q_vol * R_pellet ** 2) / (2 * R_clad)
    return q_wall


def solve_cladding(T_clad_outer, q_wall, R_pellet, R_clad, k_clad, n_nodes):
    dr = (R_clad - R_pellet) / (n_nodes - 1)

    radius = np.zeros(n_nodes)
    T = np.zeros(n_nodes)

    for i in range(n_nodes):
        radius[i] = R_pellet + i * dr
        T[i] = T_clad_outer + q_wall * R_clad / k_clad * np.log(R_clad / radius[i])


    return radius, T