import os

import numpy as np
import vtk
from vtk.util import numpy_support


def save_rod_vtk(r_c_pellet, r_c_clad, T_pellet, T_clad, nr_pellet, nr_clad, z_c, nz,
                 length_ref, T_inf, delta_T, step, folder):

    filename = os.path.join(folder, f"rod_{step:06d}.vtk")

    nr_rod = nr_pellet + nr_clad
    r_in = np.concatenate((r_c_pellet[1:nr_pellet + 1], r_c_clad[1:nr_clad + 1])) / length_ref
    z_in = z_c[1:nz + 1]
    T_in = np.concatenate((T_pellet[1:nr_pellet + 1, 1:nz + 1], T_clad[1:nr_clad + 1, 1:nz + 1]))
    theta_in = (T_in - T_inf) / delta_T

    R, Z = np.meshgrid(r_in, z_in, indexing="xy")
    points_coords = np.column_stack((R.ravel(), Z.ravel(), np.zeros(R.size)))

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nr_rod, nz, 1)

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(points_coords, deep=True))
    grid.SetPoints(points)

    vtk_theta = numpy_support.numpy_to_vtk(theta_in.T.ravel(), deep=True)
    vtk_theta.SetName("theta")
    grid.GetPointData().AddArray(vtk_theta)

    vtk_temperature = numpy_support.numpy_to_vtk(T_in.T.ravel(), deep=True)
    vtk_temperature.SetName("temperature")
    grid.GetPointData().AddArray(vtk_temperature)

    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    writer.Write()

    return filename