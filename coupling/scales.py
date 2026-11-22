import numpy as np


def compute_wall_heat_flux(q_vol, R_pellet, R_clad):
    q_wall = (q_vol * R_pellet ** 2) / (2 * R_clad)
    return q_wall


def compute_reference_scales(q_wall, R_clad, R_outer, k_fluid, g, beta):
    lenght_ref = R_outer - R_clad
    delta_T = q_wall * lenght_ref / k_fluid
    u_ref = np.sqrt(g * beta * delta_T * lenght_ref)
    return lenght_ref, delta_T, u_ref


def compute_dimensionless_numbers(lenght_ref, delta_T, u_ref, g, beta, ni, alpha):
    Pr = ni / alpha
    Gr = g * beta * delta_T * lenght_ref**3 / ni**2
    Ra = Gr * Pr
    Re = u_ref * lenght_ref / ni
    return Pr, Gr, Ra, Re


def theta_to_temperature(theta, T_inf, delta_T):
    T = T_inf + delta_T * theta
    return T

