import os
import numpy as np

import config
from octree.octree import poisson_surface_reconstruction
from utils.ply import read_ply, write_ply

if __name__ == '__main__':

    data = read_ply(config.inppath)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

    print('num of points',points.shape[0])
    node_vals = poisson_surface_reconstruction(
            config.octdepth,
            config.divtempt)(points, normals)

    write_ply(config.outpath, list(node_vals.values()), list(node_vals.keys()))
