import numpy as np

from coupling.scales import theta_to_temperature
from rod.cladding import solve_cladding
from rod.pellet import solve_constant_conductivity


def compute_cladding_wall_temperature(theta, nz, T_inf, delta_T):
    T_clad_outer = np.zeros(nz)

    for j in range(1, nz + 1):
        theta_wall = (theta[0,j] + theta[1, j]) / 2
        T_clad_outer[j - 1] = theta_to_temperature(theta_wall, T_inf, delta_T)

    return T_clad_outer


def solve_rod(T_clad_outer, nz, q_vol, q_wall, R_pellet, R_clad, k_clad, k_UO2, n_nodes_clad, n_nodes_pellet):

    n_nodes_rod = n_nodes_pellet + n_nodes_clad
    radius_rod = np.zeros(n_nodes_rod)
    T_rod = np.zeros((n_nodes_rod, nz))
    T_interface = np.zeros(nz)
    T_centre = np.zeros(nz)

    for j in range(nz):
        radius_clad, T_clad = solve_cladding(T_clad_outer[j], q_wall, R_pellet, R_clad, k_clad, n_nodes_clad)
        T_interface[j] = T_clad[0]

        radius_pellet, T_pellet, T_exact = solve_constant_conductivity(k_UO2, T_interface[j], q_vol, R_pellet, n_nodes_pellet)
        T_centre[j] = T_pellet[0]

        for i in range(n_nodes_pellet):
            radius_rod[i] = radius_pellet[i]
            T_rod[i, j] = T_pellet[i]

        for i in range(n_nodes_clad):
            radius_rod[n_nodes_pellet + i] = radius_clad[i]
            T_rod[n_nodes_pellet + i, j] = T_clad[i]

    return T_interface, T_centre, radius_rod, T_rod
