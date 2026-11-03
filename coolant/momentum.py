import numpy as np

from boundary import apply_bc_velocity


def compute_rhs_v_r(v_r, v_z, dr, dz, r_f, r_c, nr, nz, Re):
    convection_r = np.zeros((nr + 2, nz + 2))
    convection_z = np.zeros((nr + 2, nz + 2))
    diffusion_r = np.zeros((nr + 2, nz + 2))
    diffusion_z = np.zeros((nr + 2, nz + 2))
    diffusion_cyl = np.zeros((nr + 2, nz + 2))
    coeff = 1 / Re

    for i in range(1, nr):
        for j in range(1, nz + 1):

            convection_r[i, j] = (r_c[i + 1] * ((v_r[i, j] + v_r[i + 1, j]) / 2)**2
                                  - r_c[i] * ((v_r[i - 1, j] + v_r[i, j]) / 2)**2) / (r_f[i] * dr)
            convection_z[i, j] = ((v_r[i, j] + v_r[i, j + 1]) / 2 * (v_z[i, j] + v_z[i + 1, j]) / 2
                                  - (v_r[i, j - 1] + v_r[i, j]) / 2 * (v_z[i, j - 1] + v_z[i + 1, j - 1]) / 2) / dz

            diffusion_r[i, j] = (r_c[i + 1] * (v_r[i + 1, j] - v_r[i, j])
                                 - r_c[i] * (v_r[i, j] - v_r[i - 1, j])) / (r_f[i] * dr**2)
            diffusion_cyl[i, j] = - v_r[i, j] / r_f[i]**2
            diffusion_z[i, j] = (v_r[i, j + 1] - 2 * v_r[i, j] + v_r[i, j - 1]) / dz**2

    rhs_r = -(convection_r + convection_z) + coeff * (diffusion_r + diffusion_cyl + diffusion_z)

    return rhs_r


def compute_rhs_v_z(v_r, v_z, theta, dr, dz, r_f, r_c, nr, nz, Re, Gr):
    convection_r = np.zeros((nr + 2, nz + 2))
    convection_z = np.zeros((nr + 2, nz + 2))
    diffusion_r = np.zeros((nr + 2, nz + 2))
    diffusion_z = np.zeros((nr + 2, nz + 2))
    forcing = np.zeros((nr + 2, nz + 2))
    coeff = 1 / Re
    coeff_forcing = Gr / Re**2

    for i in range(1, nr + 1):
        for j in range(1, nz):

            convection_r[i, j] = (r_f[i] * (v_r[i, j] + v_r[i, j + 1]) / 2 * (v_z[i, j] + v_z[i + 1, j]) / 2
                                  - r_f[i - 1] * (v_r[i - 1, j] + v_r[i - 1, j + 1]) / 2 * (v_z[i - 1, j] + v_z[i, j]) / 2) / (r_c[i] * dr)
            convection_z[i, j] = (((v_z[i, j] + v_z[i, j + 1]) / 2)**2
                                  - ((v_z[i, j - 1] + v_z[i, j]) / 2)**2) / dz

            diffusion_r[i, j] = (r_f[i] * (v_z[i + 1, j] - v_z[i, j])
                                 - r_f[i - 1] * (v_z[i, j] - v_z[i - 1, j])) / (r_c[i] * dr**2)
            diffusion_z[i, j] = (v_z[i, j + 1] - 2 * v_z[i, j] + v_z[i, j - 1]) / dz**2

            forcing[i, j] = (theta[i, j] + theta[i, j + 1]) / 2

    rhs_z = -(convection_r + convection_z) + coeff * (diffusion_r + diffusion_z) + coeff_forcing * forcing

    return rhs_z