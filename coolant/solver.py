import os
import glob

import numpy as np

from temperature import theta_euler_step
from momentum import chorin_euler_step
from poisson import build_poisson_matrix, build_gauss_seidel_matrices
from projection import project_velocity, compute_divergence_of_v
from output_vtk import save_vtk


def run_simulation(theta, u_r, u_z, dt, nt, dr, dz, r_f, r_c, z_c, nr, nz, Re, Pr, Gr,
                   tol, max_iter, save_every, folder):

    os.makedirs(folder, exist_ok=True)
    for old_file in glob.glob(os.path.join(folder, "*.vtk")):
        os.remove(old_file)

    A = build_poisson_matrix(dr, dz, r_f, r_c, nr, nz)
    C_GS, M_inv = build_gauss_seidel_matrices(A)
    X = np.zeros(nr * nz)

    save_vtk(r_c, z_c, theta, u_r, u_z, nr, nz, 0, folder)

    for n in range(1, nt + 1):

        v_r, v_z = chorin_euler_step(u_r.copy(), u_z.copy(), theta, dt, dr, dz, r_f, r_c, nr, nz, Re, Gr)
        theta = theta_euler_step(theta, dt, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr)
        u_r, u_z, X, n_iter = project_velocity(v_r, v_z, X, C_GS, M_inv, dr, dz, r_f, r_c, nr, nz, tol, max_iter)

        div_u = compute_divergence_of_v(u_r, u_z, dr, dz, r_f, r_c, nr, nz)
        print(f"iteration {n}, time = {n * dt:.3f}, Poisson iterations = {n_iter}, "
              f"max div u = {np.max(np.abs(div_u)):.1e}")

        if n % save_every == 0:
            filename = save_vtk(r_c, z_c, theta, u_r, u_z, nr, nz, n, folder)
            print(f"saved {filename}")

    return theta, u_r, u_z