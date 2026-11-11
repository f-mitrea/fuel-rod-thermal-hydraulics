import numpy as np


def build_poisson_matrix(dr, dz, r_f, r_c, nr, nz):
    n = nr * nz
    A = np.zeros((n, n))

    for j in range(1, nz + 1):
        for i in range(1, nr + 1):

            p = (j - 1) * nr + (i - 1)

            coeff_right = r_f[i] / (r_c[i] * dr**2) if i < nr else 0.0
            coeff_left = r_f[i - 1] / (r_c[i] * dr**2) if i > 1 else 0.0
            coeff_top = 1 / dz**2 if j < nz else 0.0
            coeff_bottom = 1 / dz**2 if j > 1 else 0.0
            coeff_centre = coeff_right + coeff_left + coeff_top + coeff_bottom

            A[p, p] = - coeff_centre
            if i < nr:
                A[p, p + 1] = coeff_right
            if i > 1:
                A[p, p - 1] = coeff_left
            if j < nz:
                A[p, p + nr] = coeff_top
            if j > 1:
                A[p, p - nr] = coeff_bottom

    return A


def build_poisson_rhs(div_v, nr, nz):
    B = np.zeros(nr * nz)

    for j in range(1, nz + 1):
        for i in range(1, nr + 1):
            p = (j - 1) * nr + (i - 1)
            B[p] = - div_v[i, j]

    return B


def build_gauss_seidel_matrices(A):
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)

    M = D + L
    N = U
    M_inv = np.linalg.inv(M)
    C_GS = - M_inv @ N

    return C_GS, M_inv


def solve_poisson_gauss_seidel(X, div_v, C_GS, M_inv, nr, nz, tol, max_iter):
    B = build_poisson_rhs(div_v, nr, nz)
    Q_GS = M_inv @ B

    for k in range(1, max_iter + 1):
        X_new = C_GS @ X + Q_GS

        if np.max(np.abs(X_new - X)) < tol:
            X = X_new
            break

        X = X_new

    return X, k


def vector_to_phi(X, nr, nz):
    phi = np.zeros((nr + 2, nz + 2))

    for j in range(1, nz + 1):
        for i in range(1, nr + 1):
            p = (j - 1) * nr + (i - 1)
            phi[i, j] = X[p]

    return phi