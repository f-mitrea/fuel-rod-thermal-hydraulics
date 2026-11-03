import os
import numpy as np
import vtk
from vtk.util import numpy_support


def save_vtk(r_c, z_c, theta, step, folder):

    filename = os.path.join(folder, f"theta_{step:06d}.vtk")

    r_in = r_c[1:-1]
    z_in = z_c[1:-1]
    theta_in = theta[1:-1, 1:-1]

    n_points_r = len(r_in)
    n_points_z = len(z_in)

    R, Z = np.meshgrid(r_in, z_in, indexing="xy")

    r_flat = R.ravel()
    z_flat = Z.ravel()
    zeros = np.zeros_like(r_flat)

    points_coords = np.column_stack((r_flat, z_flat, zeros))

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(n_points_r, n_points_z, 1)

    vtk_array = numpy_support.numpy_to_vtk(points_coords, deep=True)
    points = vtk.vtkPoints()
    points.SetData(vtk_array)
    grid.SetPoints(points)

    theta_flat = theta_in.T.ravel()
    vtk_array_theta = numpy_support.numpy_to_vtk(theta_flat, deep=True)
    vtk_array_theta.SetName("theta")
    grid.GetPointData().AddArray(vtk_array_theta)

    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    writer.Write()

    return filename