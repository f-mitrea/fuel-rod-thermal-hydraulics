import os

import numpy as np
import vtk
from vtk.util import numpy_support


def save_rod_vtk(radius_rod, z_c, T_rod, nz, length_ref, T_inf, delta_T, folder):

    filename = os.path.join(folder, "rod.vtk")

    n_nodes_rod = len(radius_rod)
    r_in = radius_rod / length_ref 
    z_in = z_c[1:nz + 1] 
    theta_rod = (T_rod - T_inf) / delta_T 

    R, Z = np.meshgrid(r_in, z_in, indexing="xy") 
    points_coords = np.column_stack((R.ravel(), Z.ravel(), np.zeros(R.size))) 

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(n_nodes_rod, nz, 1)

    points = vtk.vtkPoints()
    points.SetData(numpy_support.numpy_to_vtk(points_coords, deep=True))
    grid.SetPoints(points)

    vtk_theta = numpy_support.numpy_to_vtk(theta_rod.T.ravel(), deep=True)
    vtk_theta.SetName("theta")
    grid.GetPointData().AddArray(vtk_theta)

    vtk_temperature = numpy_support.numpy_to_vtk(T_rod.T.ravel(), deep=True)
    vtk_temperature.SetName("temperature")
    grid.GetPointData().AddArray(vtk_temperature)

    writer = vtk.vtkStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.SetFileTypeToBinary()
    writer.Write()

    return filename