import numpy as np
from itertools import product

class plot:

    def __init__(self, func):
        self.func = func
        


    def create_mesh(self):

        x = np.arange(-5, 5, 0.025)
        y = np.arange(-5, 5, 0.025)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)

        mesh_size = range(len(X))

        for i, j in product(mesh_size, mesh_size):
            x_coor = X[i][j]
            y_coor = Y[i][j]
            Z[i][j] = self.func(np.array([x_coor, y_coor]))

        return X, Y, Z
    
