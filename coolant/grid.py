import numpy as np

def make_grid(r_min, r_max, z_max, nr, nz):
    dr = (r_max - r_min) / nr
    dz = z_max / nz
    r_c = r_min - dr / 2 + dr * np.arange(nr + 2)
    z_c = - dz / 2 + dz * np.arange(nz + 2)
    r_f = r_c + dr / 2
    z_f = z_c + dz / 2
    return dr,dz, r_c, z_c, r_f, z_f
