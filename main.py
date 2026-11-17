import numpy as np

from coolant.grid import make_grid
from coolant.solver import run_simulation
from rod.cladding import compute_wall_heat_flux
from coupling.scales import compute_reference_scales, compute_dimensionless_numbers
from coupling.coupling import compute_cladding_wall_temperature, solve_rod
from coupling.output_vtk import save_rod_vtk


# Fuel rod (dimensional, SI units)
R_pellet = 4.0e-3
R_clad = 4.6e-3
q_vol = 2.75e6
k_clad = 16.0
k_UO2 = 2.3
n_nodes_pellet = 100
n_nodes_clad = 20

# Coolant (dimensional, SI units)
R_outer = 9.2e-3
height = 4.6e-3
T_inf = 560.0
k_fluid = 0.6
g = 9.81
beta = 2.0e-4
nu = 7.0e-7
alpha = 1.0e-6

# Numerical parameters of the coolant solver (dimensionless)
nr, nz = 32, 8
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

theta = np.zeros((nr + 2, nz + 2))
u_r = np.zeros((nr + 2, nz + 2))
u_z = np.zeros((nr + 2, nz + 2))

theta, u_r, u_z = run_simulation(theta, u_r, u_z, dt, nt, dr, dz, r_f, r_c, z_c, nr, nz, Re, Pr, Gr, tol, max_iter, save_every, folder)

T_clad_outer = compute_cladding_wall_temperature(theta, nz, T_inf, delta_T)

T_interface, T_centre, radius_rod, T_rod = solve_rod(T_clad_outer, nz, q_vol, q_wall, R_pellet, R_clad, k_clad, k_UO2, n_nodes_clad, n_nodes_pellet)

filename = save_rod_vtk(radius_rod, z_c, T_rod, nz, length_ref, T_inf, delta_T, folder)
print(f"saved {filename}")

print("\n   z [mm]   T cladding outer [K]   T pellet surface [K]   T centre [K]")
for j in range(nz):
    print(f"   {z_c[j + 1] * length_ref * 1e3:6.3f}   {T_clad_outer[j]:20.4f}   {T_interface[j]:20.4f}   {T_centre[j]:12.4f}")