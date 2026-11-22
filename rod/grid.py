import numpy as np


def make_radial_grid(r_min, r_max, nr):
    dr = (r_max - r_min) / nr
    r_c = r_min - dr / 2 + dr * np.arange(nr + 2)
    r_f = r_c + dr / 2
    return dr, r_c, r_f