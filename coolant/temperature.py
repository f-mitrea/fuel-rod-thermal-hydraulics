import os
import glob

import numpy as np

from boundary import apply_bc
from output_vtk import save_vtk


def compute_rhs(theta, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr):
    convection_z = np.zeros((nr + 2, nz + 2))
    convection_r = np.zeros((nr + 2, nz + 2))
    diffusion_z = np.zeros((nr + 2, nz + 2))
    diffusion_r = np.zeros((nr + 2, nz + 2))
    coeff = 1/(Re * Pr)
    
    for i in range(1, nr + 1):
        for j in range(1, nz + 1):
            
            convection_z[i, j] = ((theta[i, j + 1] + theta[i , j]) / 2 * u_z[i , j] - (theta[i, j] + theta[i , j - 1]) / 2 * u_z[i , j - 1]) / dz
            convection_r[i, j] = ((r_f[i] * (theta[i + 1, j] + theta[i, j]) / 2) * u_r[i , j] - (r_f[i - 1] * (theta[i, j] + theta[i - 1, j]) / 2) * u_r[i - 1 , j]) / ( r_c[i] * dr )
            
            diffusion_z[i, j] = (theta[i, j + 1] - 2 * theta[i,j] + theta[i, j - 1]) / dz**2
            diffusion_r[i, j] = ((r_f[i] * (theta[i + 1, j] - theta[i, j])) - (r_f[i - 1] * (theta[i, j] - theta[i - 1, j]))) / (r_c[i] * dr**2)

    rhs = -(convection_r + convection_z) + coeff * (diffusion_z + diffusion_r)

    return rhs


def theta_euler(theta, dt, nt, u_r, u_z, dr, dz, r_f, r_c, z_c, nr, nz, Re, Pr, save_every, folder):

    os.makedirs(folder, exist_ok=True)
    for old_file in glob.glob(os.path.join(folder, "theta_*.vtk")): os.remove(old_file)
    save_vtk(r_c, z_c, theta, 0, folder)

    for n in range(1, nt + 1):

        theta = apply_bc(theta, nr, nz)
        rhs = compute_rhs(theta, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr)

        theta[1:nr+1, 1:nz+1] += dt * rhs[1:nr+1, 1:nz+1]

        t = n * dt
        print(f"iteration {n}, time = {t:.3f}")

        if n % save_every == 0:
            filename = save_vtk(r_c, z_c, theta, n, folder)
            print(f"saved {filename}")

    theta = apply_bc(theta, nr, nz)

    return theta