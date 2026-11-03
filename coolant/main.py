import numpy as np
from grid import make_grid
from temperature import theta_euler


r_min, r_max, z_max = 1.0, 2.0, 1.0
nr, nz = 32, 8
Re, Pr = 20.0, 1.0
dt = 0.001
nt = 30000

dr, dz, r_c, z_c, r_f, z_f = make_grid(r_min, r_max, z_max, nr, nz)
theta = np.zeros((nr + 2, nz + 2))
u_r = np.zeros((nr + 2, nz + 2))
u_z = np.zeros((nr + 2, nz + 2))

theta = theta_euler(theta, dt, nt, u_r, u_z, dr, dz, r_f, r_c, z_c, nr, nz, Re, Pr, 3000, "output")