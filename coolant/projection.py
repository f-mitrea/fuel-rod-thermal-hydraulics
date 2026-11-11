import numpy as np

from poisson import solve_poisson_gauss_seidel, vector_to_phi


def compute_divergence_of_v(v_r, v_z, dr, dz, r_f, r_c, nr, nz):
    div_v = np.zeros((nr + 2, nz + 2))

    for i in range(1, nr + 1):
        for j in range(1, nz + 1):
            div_v[i, j] = ((r_f[i] * v_r[i, j] - r_f[i - 1] * v_r[i - 1, j]) / (r_c[i] * dr)
                           + (v_z[i, j] - v_z[i, j - 1]) / dz)

    return div_v


def correct_velocity(v_r, v_z, phi, dr, dz, nr, nz):
    u_r = v_r.copy()
    u_z = v_z.copy()

    for i in range(1, nr):
        for j in range(1, nz + 1):
            u_r[i, j] = v_r[i, j] + (phi[i + 1, j] - phi[i, j]) / dr

    for i in range(1, nr + 1):
        for j in range(1, nz):
            u_z[i, j] = v_z[i, j] + (phi[i, j + 1] - phi[i, j]) / dz

    return u_r, u_z


def project_velocity(v_r, v_z, X, C_GS, M_inv, dr, dz, r_f, r_c, nr, nz, tol, max_iter):
    div_v = compute_divergence_of_v(v_r, v_z, dr, dz, r_f, r_c, nr, nz)
    X, n_iter = solve_poisson_gauss_seidel(X, div_v, C_GS, M_inv, nr, nz, tol, max_iter)
    phi = vector_to_phi(X, nr, nz)
    u_r, u_z = correct_velocity(v_r, v_z, phi, dr, dz, nr, nz)
    return u_r, u_z, X, n_iter