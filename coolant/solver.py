import os
import glob

import numpy as np

from coolant.temperature import theta_euler_step
from coolant.momentum import chorin_euler_step
from coolant.poisson import build_poisson_matrix, build_gauss_seidel_matrices
from coolant.projection import project_velocity, compute_divergence_of_v
from coolant.output_vtk import save_vtk
from rod.temperature import rod_euler_step
from coupling.interface import compute_wall_theta, compute_fluid_wall_flux
from coupling.output_vtk import save_rod_vtk


def run_simulation(theta, u_r, u_z, dt, nt, dr, dz, r_f, r_c, z_c, nr, nz, Re, Pr, Gr,
                   tol, max_iter, save_every, folder,
                   T_pellet, T_clad, dt_rod, dz_rod,
                   r_c_pellet, r_f_pellet, dr_pellet, nr_pellet, k_pellet, rho_c_pellet,
                   r_c_clad, r_f_clad, dr_clad, nr_clad, k_clad, rho_c_clad, q_vol,
                   T_inf, delta_T, length_ref, k_fluid):

    os.makedirs(folder, exist_ok=True)
    for old_file in glob.glob(os.path.join(folder, "*.vtk")):
        os.remove(old_file)

    A = build_poisson_matrix(dr, dz, r_f, r_c, nr, nz)
    C_GS, M_inv = build_gauss_seidel_matrices(A)
    X = np.zeros(nr * nz)

    save_vtk(r_c, z_c, theta, u_r, u_z, nr, nz, 0, folder)
    save_rod_vtk(r_c_pellet, r_c_clad, T_pellet, T_clad, nr_pellet, nr_clad, z_c, nz,
                 length_ref, T_inf, delta_T, 0, folder)

    for n in range(1, nt + 1):

        v_r, v_z = chorin_euler_step(u_r.copy(), u_z.copy(), theta, dt, dr, dz, r_f, r_c, nr, nz, Re, Gr)

        theta_wall = compute_wall_theta(T_clad, nr_clad, nz, T_inf, delta_T)
        q_wall = compute_fluid_wall_flux(theta, theta_wall, nz, dr, k_fluid, delta_T, length_ref)

        theta = theta_euler_step(theta, theta_wall, dt, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr)
        T_pellet, T_clad = rod_euler_step(T_pellet, T_clad, q_wall, dt_rod, dz_rod, nz,
                                          r_c_pellet, r_f_pellet, dr_pellet, nr_pellet, k_pellet, rho_c_pellet,
                                          r_c_clad, r_f_clad, dr_clad, nr_clad, k_clad, rho_c_clad, q_vol)

        u_r, u_z, X, n_iter = project_velocity(v_r, v_z, X, C_GS, M_inv, dr, dz, r_f, r_c, nr, nz, tol, max_iter)

        div_u = compute_divergence_of_v(u_r, u_z, dr, dz, r_f, r_c, nr, nz)
        print(f"iteration {n}, time = {n * dt:.3f}, Poisson iterations = {n_iter}, "
              f"max div u = {np.max(np.abs(div_u)):.1e}, "
              f"mean wall heat flux = {np.mean(q_wall[1:nz + 1]):.1f} W/m^2, "
              f"max pellet temperature = {np.max(T_pellet[1:nr_pellet + 1, 1:nz + 1]):.3f} K")

        if n % save_every == 0:
            filename = save_vtk(r_c, z_c, theta, u_r, u_z, nr, nz, n, folder)
            print(f"saved {filename}")
            filename = save_rod_vtk(r_c_pellet, r_c_clad, T_pellet, T_clad, nr_pellet, nr_clad, z_c, nz,
                                    length_ref, T_inf, delta_T, n, folder)
            print(f"saved {filename}")

    return theta, u_r, u_z, T_pellet, T_clad