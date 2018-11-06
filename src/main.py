import os
import numpy as np

import config
from octree.octree import poisson_surface_reconstruction
from utils.ply import read_ply, write_ply

if __name__ == '__main__':

    data = read_ply(config.inppath)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

    print(points.shape)
    print(normals.shape)
    node_locs, node_grads = poisson_surface_reconstruction(config.octdepth)(points, normals)

    write_ply(config.outpath, (node_locs, node_grads), ['x', 'y', 'z', 'nx', 'ny', 'nz'])
