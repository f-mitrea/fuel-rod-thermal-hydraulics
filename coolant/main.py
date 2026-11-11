import numpy as np

from grid import make_grid
from solver import run_simulation


r_min, r_max, z_max = 1.0, 2.0, 1.0
nr, nz = 32, 8

Ra, Pr = 1.0e4, 0.7
Gr = Ra / Pr
Re = np.sqrt(Gr)

dt = 0.005
nt = 20000
tol = 1.0e-12
max_iter = 100000
save_every = 500
folder = "output"

dr, dz, r_c, z_c, r_f, z_f = make_grid(r_min, r_max, z_max, nr, nz)

theta = np.zeros((nr + 2, nz + 2))
u_r = np.zeros((nr + 2, nz + 2))
u_z = np.zeros((nr + 2, nz + 2))

theta, u_r, u_z = run_simulation(theta, u_r, u_z, dt, nt, dr, dz, r_f, r_c, z_c, nr, nz, Re, Pr, Gr,
                                 tol, max_iter, save_every, folder)