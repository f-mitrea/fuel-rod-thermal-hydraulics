def apply_bc(theta, nr, nz):
    
    for j in range(1, nz + 1):
        theta[0, j] = 2 - theta[1, j]
        theta[nr + 1, j] = - theta[nr, j]

    for  i in range(1, nr + 1):
        theta[i, 0] = theta[i, 1]
        theta[i, nz + 1] = theta[i, nz]

    return theta

def apply_bc_velocity(v_r, v_z, nr, nz):

    for j in range(nz + 2):
        v_r[0, j] = 0.0
        v_r[nr, j] = 0.0

    for i in range(1, nr):
        v_r[i, 0] = - v_r[i, 1]
        v_r[i, nz + 1] = - v_r[i, nz]

    for i in range(nr + 2):
        v_z[i, 0] = 0.0
        v_z[i, nz] = 0.0

    for j in range(1, nz):
        v_z[0, j] = - v_z[1, j]
        v_z[nr + 1, j] = - v_z[nr, j]

    return v_r, v_z