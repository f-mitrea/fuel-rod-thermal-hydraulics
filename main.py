import numpy as np

from coolant.grid import make_grid
from coolant.solver import run_simulation
from rod.grid import make_radial_grid
from coupling.scales import compute_wall_heat_flux, compute_reference_scales, compute_dimensionless_numbers

# rod_data
R_pellet = 4.0e-3
R_clad = 4.6e-3
q_vol = 2.75e6
k_clad = 16.0
k_UO2 = 2.3
rho_c_UO2 = 10970.0 * 300.0
rho_c_clad = 6550.0 * 330.0

# coolant_data
R_outer = 9.2e-3
height = 4.6e-3
T_inf = 560.0
k_fluid = 0.6
g = 9.81
beta = 2.0e-4
nu = 7.0e-7
alpha = 1.0e-6

# numeric_data
nr, nz = 32, 8
nr_pellet = 20
nr_clad = 3
dt = 0.005
nt = 30000
tol = 1.0e-12
max_iter = 100000
save_every = 500
folder = "output"

q_wall = compute_wall_heat_flux(q_vol, R_pellet, R_clad)

length_ref, delta_T, u_ref = compute_reference_scales(q_wall, R_clad, R_outer, k_fluid, g, beta)
Pr, Gr, Ra, Re = compute_dimensionless_numbers(length_ref, delta_T, u_ref, g, beta, nu, alpha)

r_min = R_clad / length_ref
r_max = R_outer / length_ref
z_max = height / length_ref

print(f"wall heat flux = {q_wall:.4e} W/m^2")
print(f"reference length = {length_ref:.4e} m, reference temperature difference = {delta_T:.4f} K, "
      f"reference velocity = {u_ref:.4e} m/s")
print(f"Pr = {Pr:.4f}, Gr = {Gr:.4e}, Ra = {Ra:.4e}, Re = {Re:.4f}")

dr, dz, r_c, z_c, r_f, z_f = make_grid(r_min, r_max, z_max, nr, nz)

dz_rod = dz * length_ref
z_c_rod = z_c * length_ref
dt_rod = dt * length_ref / u_ref

dr_pellet, r_c_pellet, r_f_pellet = make_radial_grid(0.0, R_pellet, nr_pellet)
dr_clad, r_c_clad, r_f_clad = make_radial_grid(R_pellet, R_clad, nr_clad)

theta = np.zeros((nr + 2, nz + 2))
u_r = np.zeros((nr + 2, nz + 2))
u_z = np.zeros((nr + 2, nz + 2))

T_pellet = np.full((nr_pellet + 2, nz + 2), T_inf)
T_clad = np.full((nr_clad + 2, nz + 2), T_inf)

theta, u_r, u_z, T_pellet, T_clad = run_simulation(theta, u_r, u_z, dt, nt, dr, dz, r_f, r_c, z_c, nr, nz, Re, Pr, Gr,
                                                   tol, max_iter, save_every, folder,
                                                   T_pellet, T_clad, dt_rod, dz_rod,
                                                   r_c_pellet, r_f_pellet, dr_pellet, nr_pellet, k_UO2, rho_c_UO2,
                                                   r_c_clad, r_f_clad, dr_clad, nr_clad, k_clad, rho_c_clad, q_vol,
                                                   T_inf, delta_T, length_ref, k_fluid)