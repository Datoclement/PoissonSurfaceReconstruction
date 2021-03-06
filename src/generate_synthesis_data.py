import os, sys
import numpy as np
import argparse

import config
from utils.ply import write_ply

def generate_cube(side, sample_per_side, out):

    sps = sample_per_side
    arr = np.arange(sample_per_side**3)
    x = arr % sps
    y = arr // sps % sps
    z = arr // sps // sps % sps

    coords = np.vstack([x,y,z]).T
    is_boundary = np.logical_or.reduce(np.logical_or(coords==0, coords==sps-1), axis=-1)
    coords = coords[is_boundary].astype(np.float32)

    write_ply(out, (coords,), ['x', 'y', 'z'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--side", default=1., type=float, help="size of the cube in absolute value")
    parser.add_argument("--sps", default=51, type=int, help="number of points per side")
    parser.add_argument("--out", default=os.path.join(config.outdir, 'cube.ply'), type=str, help="path for output file")
    args = parser.parse_args()

    generate_cube(side=args.side, sample_per_side=args.sps, out='_'.join([args.out, str(args.sps)]))
