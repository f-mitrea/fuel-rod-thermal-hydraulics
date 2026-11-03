def apply_bc(theta, nr, nz):
    
    for j in range(1, nz + 1):
        theta[0, j] = 2 - theta[1, j]
        theta[nr + 1, j] = - theta[nr, j]

    for  i in range(1, nr + 1):
        theta[i, 0] = theta[i, 1]
        theta[i, nz + 1] = theta[i, nz]

    return theta