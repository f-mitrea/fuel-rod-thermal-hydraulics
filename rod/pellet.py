import numpy as np

from rod.linear_solvers import solve_tridiagonal_system


def assemble_constant_conductivity_system(
    p_coeff, q_coeff, r_coeff, dr, n_nodes, T_wall, q_vol, k_UO2
):

    A = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)
    gamma = q_vol * dr**2 / k_UO2

    for i in range(n_nodes):
        if i > 0 and i < n_nodes - 1:
            b[i] = dr**2 * r_coeff[i]
            A[i, i] = 2 + dr**2 * q_coeff[i]
            A[i, i + 1] = -1 + dr / 2 * p_coeff[i]
            A[i, i - 1] = -1 - dr / 2 * p_coeff[i]
        elif i == 0:
            b[i] = gamma
            A[i, i] = 4
            A[i, i + 1] = -4
        elif i == n_nodes - 1:
            b[i] = dr**2 * r_coeff[i] + (1 - dr / 2 * p_coeff[i]) * T_wall
            A[i, i] = 2 + dr**2 * q_coeff[i]
            A[i, i - 1] = -1 - dr / 2 * p_coeff[i]
    return A, b


def solve_constant_conductivity(
    k_UO2, T_wall, q_vol, R_pellet, n_nodes
):

    dr = R_pellet / n_nodes

    radius = np.zeros(n_nodes)
    for i in range(n_nodes):
        radius[i] = i * dr

    p_coeff = np.zeros(n_nodes)
    p_coeff[1:] = -1.0 / radius[1:]

    r_coeff = q_vol / k_UO2 * np.ones(n_nodes)
    q_coeff = np.zeros(n_nodes)

    A, b = assemble_constant_conductivity_system(
        p_coeff, q_coeff, r_coeff, dr, n_nodes, T_wall, q_vol, k_UO2
    )

    T_num = solve_tridiagonal_system(A, b)

    radius = np.append(radius, R_pellet)
    T_num = np.append(T_num, T_wall)

    T_exact = T_wall + q_vol * (R_pellet**2 - radius**2) / (4 * k_UO2)

    return radius, T_num, T_exact