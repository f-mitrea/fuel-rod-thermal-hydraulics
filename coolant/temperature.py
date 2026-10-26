import os

import numpy as np

def make_grid(r_min, r_max, z_max, nr, nz):
    dr = (r_max - r_min) / nr
    dz = z_max / nz
    r_c = r_min - dr / 2 + dr * np.arange(nr + 2)
    z_c = - dz / 2 + dz * np.arange(nz + 2)
    r_f = r_c + dr / 2
    z_f = z_c + dz / 2
    return dr,dz, r_c, z_c, r_f, z_f


def apply_bc(theta, nr, nz):
    
    for j in range(1, nz + 1):
        theta[0, j] = 2 - theta[1, j]
        theta[nr + 1, j] = - theta[nr, j]

    for  i in range(1, nr + 1):
        theta[i, 0] = theta[i, 1]
        theta[i, nz + 1] = theta[i, nz]

    return theta



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


def start_file(file_path, nr, nz, dt):
    with open(file_path, "w") as f:
        f.write("# Temperature theta in the coolant (dimensionless)\n")
        f.write("# rows: r, from the cladding outwards | columns: z, from the bottom up\n")
        f.write(f"# nr = {nr}, nz = {nz}, dt = {dt}\n")
        f.write("\n")


def save_theta(theta, n, dt, nr, nz, file_path):
    with open(file_path, "a") as f:
        f.write(f"# iteration {n}    time {n * dt:.4f}\n")
        np.savetxt(f, theta[1:nr+1, 1:nz+1], fmt="%10.6f")
        f.write("\n")


def theta_euler(theta, dt, nt, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr, save_every, folder):

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "temperature_evolution.txt")
    start_file(file_path, nr, nz, dt)
    save_theta(theta, 0, dt, nr, nz, file_path)

    for n in range(1, nt + 1):
        
        theta = apply_bc(theta, nr, nz)
        rhs = compute_rhs(theta, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr)

        theta[1:nr+1, 1:nz+1] += dt * rhs[1:nr+1, 1:nz+1]

        if n % save_every == 0:
            save_theta(theta, n, dt, nr, nz, file_path)

    theta = apply_bc(theta, nr, nz)

    return theta


if __name__ == "__main__":
    r_min, r_max, z_max = 1.0, 2.0, 1.0
    nr, nz = 32, 8
    Re, Pr = 20.0, 1.0
    dt = 0.001
    nt = 30000

    dr, dz, r_c, z_c, r_f, z_f = make_grid(r_min, r_max, z_max, nr, nz)
    theta = np.zeros((nr + 2, nz + 2))
    u_r = np.zeros((nr + 2, nz + 2))
    u_z = np.zeros((nr + 2, nz + 2))

    theta = theta_euler(theta, dt, nt, u_r, u_z, dr, dz, r_f, r_c, nr, nz, Re, Pr, 3000, "temperature_evolution")