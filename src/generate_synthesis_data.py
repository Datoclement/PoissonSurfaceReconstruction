import os
import numpy as np

import config
from utils.ply import write_ply

def generate_cube(side=1., sample_per_side=51):

    sps = sample_per_side
    arr = np.arange(sample_per_side**3)
    x = arr % sps
    y = arr // sps % sps
    z = arr // sps // sps % sps

    coords = np.vstack([x,y,z]).T
    is_boundary = np.logical_or.reduce(np.logical_or(coords==0, coords==sps-1), axis=-1)
    coords = coords[is_boundary].astype(np.float32)

    write_ply(os.path.join(config.outdir, 'cube.ply'), (coords,), ['x', 'y', 'z'])


if __name__ == '__main__':
    generate_cube()
