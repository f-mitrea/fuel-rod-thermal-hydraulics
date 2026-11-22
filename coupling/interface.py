import numpy as np


def compute_wall_theta(T_clad, nr_clad, nz, T_inf, delta_T):
    theta_wall = np.zeros(nz + 2)

    for j in range(1, nz + 1):
        theta_wall[j] = (T_clad[nr_clad, j] - T_inf) / delta_T

    return theta_wall


def compute_fluid_wall_flux(theta, theta_wall, nz, dr, k_fluid, delta_T, length_ref):
    q_wall = np.zeros(nz + 2)

    for j in range(1, nz + 1):
        q_wall[j] = 2 * k_fluid * delta_T * (theta_wall[j] - theta[1, j]) / (length_ref * dr)

    return q_wall