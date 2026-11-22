import numpy as np

from rod.boundary import compute_interface_flux, apply_bc_pellet, apply_bc_clad


def compute_rhs_rod(T, r_c, r_f, dr, dz, nr, nz, k, rho_c, q_vol):
    diffusion_r = np.zeros((nr + 2, nz + 2))
    diffusion_z = np.zeros((nr + 2, nz + 2))

    for i in range(1, nr + 1):
        for j in range(1, nz + 1):

            diffusion_r[i, j] = (r_f[i] * (T[i + 1, j] - T[i, j])
                                 - r_f[i - 1] * (T[i, j] - T[i - 1, j])) / (r_c[i] * dr**2)
            diffusion_z[i, j] = (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dz**2

    rhs_T = (k * (diffusion_r + diffusion_z) + q_vol) / rho_c

    return rhs_T


def rod_euler_step(T_pellet, T_clad, q_wall, dt, dz, nz,
                   r_c_pellet, r_f_pellet, dr_pellet, nr_pellet, k_pellet, rho_c_pellet,
                   r_c_clad, r_f_clad, dr_clad, nr_clad, k_clad, rho_c_clad, q_vol):

    q_interface = compute_interface_flux(T_pellet, T_clad, nr_pellet, nz, dr_pellet, dr_clad, k_pellet, k_clad)

    T_pellet = apply_bc_pellet(T_pellet, nr_pellet, nz, dr_pellet, k_pellet, q_interface)
    T_clad = apply_bc_clad(T_clad, nr_clad, nz, dr_clad, k_clad, q_interface, q_wall)

    rhs_pellet = compute_rhs_rod(T_pellet, r_c_pellet, r_f_pellet, dr_pellet, dz, nr_pellet, nz, k_pellet, rho_c_pellet, q_vol)
    rhs_clad = compute_rhs_rod(T_clad, r_c_clad, r_f_clad, dr_clad, dz, nr_clad, nz, k_clad, rho_c_clad, 0.0)

    T_pellet[1:nr_pellet + 1, 1:nz + 1] += dt * rhs_pellet[1:nr_pellet + 1, 1:nz + 1]
    T_clad[1:nr_clad + 1, 1:nz + 1] += dt * rhs_clad[1:nr_clad + 1, 1:nz + 1]

    return T_pellet, T_clad