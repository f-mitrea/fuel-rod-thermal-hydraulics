import os

import numpy as np
import vtk
from vtk.util import numpy_support


def save_vtk(r_c, z_c, theta, u_r, u_z, nr, nz, step, folder):

    filename = os.path.join(folder, f"fields_{step:06d}.vtk")

    r_in = r_c[1:nr + 1]
    z_in = z_c[1:nz + 1]
    theta_in = theta[1:nr + 1, 1:nz + 1]

    u_r_centre = np.zeros((nr, nz))
    u_z_centre = np.zeros((nr, nz))
    for i in range(1, nr + 1):
        for j in range(1, nz + 1):
            u_r_centre[i - 1, j - 1] = (u_r[i - 1, j] + u_r[i, j]) / 2
            u_z_centre[i - 1, j - 1] = (u_z[i, j - 1] + u_z[i, j]) / 2

    R, Z = np.meshgrid(r_in, z_in, indexing="xy")
    points_coords = np.column_stack((R.ravel(), Z.ravel(), np.zeros(R.size)))

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(nr, nz, 1)

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(points_coords, deep=True))
    grid.SetPoints(points)

    vtk_theta = numpy_support.numpy_to_vtk(theta_in.T.ravel(), deep=True)
    vtk_theta.SetName("theta")
    grid.GetPointData().AddArray(vtk_theta)

    velocity = np.column_stack((u_r_centre.T.ravel(), u_z_centre.T.ravel(), np.zeros(nr * nz)))
    vtk_velocity = numpy_support.numpy_to_vtk(velocity, deep=True)
    vtk_velocity.SetName("velocity")
    grid.GetPointData().AddArray(vtk_velocity)

    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    writer.Write()

    return filename