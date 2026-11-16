import numpy as np

from coolant.boundary import apply_bc_theta


def compute_rhs_theta(theta, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr):
    convection_r = np.zeros((nr + 2, nz + 2))
    convection_z = np.zeros((nr + 2, nz + 2))
    diffusion_r = np.zeros((nr + 2, nz + 2))
    diffusion_z = np.zeros((nr + 2, nz + 2))
    coeff = 1 / (Re * Pr)

    for i in range(1, nr + 1):
        for j in range(1, nz + 1):

            convection_r[i, j] = (r_f[i] * (theta[i + 1, j] + theta[i, j]) / 2 * u_r[i, j]
                                  - r_f[i - 1] * (theta[i, j] + theta[i - 1, j]) / 2 * u_r[i - 1, j]) / (r_c[i] * dr)
            convection_z[i, j] = ((theta[i, j + 1] + theta[i, j]) / 2 * u_z[i, j]
                                  - (theta[i, j] + theta[i, j - 1]) / 2 * u_z[i, j - 1]) / dz

            diffusion_r[i, j] = (r_f[i] * (theta[i + 1, j] - theta[i, j])
                                 - r_f[i - 1] * (theta[i, j] - theta[i - 1, j])) / (r_c[i] * dr**2)
            diffusion_z[i, j] = (theta[i, j + 1] - 2 * theta[i, j] + theta[i, j - 1]) / dz**2

    rhs_theta = -(convection_r + convection_z) + coeff * (diffusion_r + diffusion_z)

    return rhs_theta


def theta_euler_step(theta, dt, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr):
    theta = apply_bc_theta(theta, nr, nz, dr)

    rhs_theta = compute_rhs_theta(theta, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr)

    theta[1:nr + 1, 1:nz + 1] += dt * rhs_theta[1:nr + 1, 1:nz + 1]
    theta = apply_bc_theta(theta, nr, nz, dr)

    return theta