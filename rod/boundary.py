import numpy as np


def compute_interface_flux(T_inner, T_outer, nr_inner, nz, dr_inner, dr_outer, k_inner, k_outer):
    q_interface = np.zeros(nz + 2)
    resistance = dr_inner / (2 * k_inner) + dr_outer / (2 * k_outer)

    for j in range(1, nz + 1):
        q_interface[j] = (T_inner[nr_inner, j] - T_outer[1, j]) / resistance

    return q_interface


def apply_bc_pellet(T_pellet, nr_pellet, nz, dr_pellet, k_pellet, q_interface):

    for j in range(1, nz + 1):
        T_pellet[0, j] = T_pellet[1, j]
        T_pellet[nr_pellet + 1, j] = T_pellet[nr_pellet, j] - dr_pellet * q_interface[j] / k_pellet

    for i in range(1, nr_pellet + 1):
        T_pellet[i, 0] = T_pellet[i, 1]
        T_pellet[i, nz + 1] = T_pellet[i, nz]

    return T_pellet


def apply_bc_clad(T_clad, nr_clad, nz, dr_clad, k_clad, q_inner, q_outer):

    for j in range(1, nz + 1):
        T_clad[0, j] = T_clad[1, j] + dr_clad * q_inner[j] / k_clad
        T_clad[nr_clad + 1, j] = T_clad[nr_clad, j] - dr_clad * q_outer[j] / k_clad

    for i in range(1, nr_clad + 1):
        T_clad[i, 0] = T_clad[i, 1]
        T_clad[i, nz + 1] = T_clad[i, nz]

    return T_clad