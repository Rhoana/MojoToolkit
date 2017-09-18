#!/usr/bin/python
import os
import cv2
import h5py
import glob
import bisect
import argparse
import numpy as np

help = {
    'in': 'input file (default in.h5)',
    'out': 'output file (default out.h5)',
    'num': 'Downsampling times in XY (default 0)',
    'numz': 'Downsampling times in Z (default 0)',
    'minz': 'Min scaled Z sections deep (default 1)',
    'f': 'Save full size, downsample for filtering',
    'hd2hd': 'Resize a hdf5 file with filtered IDs!',
}
paths = {}

parser = argparse.ArgumentParser(description=help['hd2hd'])
parser.add_argument('in', default='in.h5', nargs='?', help=help['in'])
parser.add_argument('out', default='out.h5', nargs='?', help=help['out'])
parser.add_argument('--num', '-n', default=0, type=int, help=help['num'])
parser.add_argument('--numz', '-z', default=0, type=int, help=help['numz'])
parser.add_argument('--minz', '-m', type=int, default=0, help=help['minz'])
parser.add_argument('-f', action='store_true', default=False, help=help['f'])

args = vars(parser.parse_args())
# Parse all path arguments 
for key in ['in', 'out']:
    paths[key] = os.path.realpath(os.path.expanduser(args[key]))
# Parse all integer arguments
scales = 2**np.uint32([args['numz'], args['num'], args['num']])
zmin = args['minz']
# Boolean flag
save_full = args['f']

def progress(name, n, whole, part):
    """
    Arguments
    ----------
    step: str
        Name of this progress
    n : int
        Number of steps to print
    whole : int
        Total number of interations
    part : int
        Current iteration
    """
    # simple progress indicator
    if part % (whole / n) == 1:
        msg = '{}: {:.2f}%'
        percent = 100.0 * part / whole
        print(msg.format(name, percent))

# Open the input file
with h5py.File(paths['in'],'r') as hi:

    # Get everything for first group
    group = hi.keys()[0]
    reading = hi[group]
    # Get the new shape and dtype
    dtype = reading.dtype
    in_shape = reading.shape
    scale_shape = reading.shape/scales
    # Give scales small variable names
    scale_yx = scale_shape[1:]
    sz, sy, sx = scales.tolist()

    # Make set of good ids
    good_ids = set()
    # If filtering by depth
    if zmin > 1:
        n_yx = scale_yx.prod()
        # Iterate through all image columns
        for i in range(n_yx):
            progress('read', 1000, n_yx, i)
            # Get one full-depth column
            yi, xi = np.unravel_index(i, scale_yx)
            column = reading[::sz, yi*sy, xi*sx]
            # Only measure nonzero columns
            col_max = np.amax(column)
            if col_max == 0:
                continue
            # Check if column max is too big
            if col_max > 2**32:
                # Get unique elements and counts
                all_ids, all_counts = np.unique(column, return_counts = True)
                new_ids = all_ids[all_counts >= zmin]
            else: 
                # Get all element counts
                new_ids = np.argwhere(np.bincount(column) >= zmin)[0]
            # Add to set of good ids
            good_ids |= set(new_ids.tolist())
    
    print good_ids
        
    # Check whether to save scaled or full
    out_shape = scale_shape
    if save_full:
        sz, sy, sx = [1,1,1]
        out_shape = in_shape

    # open an output file
    with h5py.File(paths['out'], 'w') as hf:
        # Write a new dataset
        written = hf.create_dataset(group, out_shape, dtype=dtype)
        # Write each Z section to output
        for zi in range(out_shape[0]):
            # Write as a scaled image from input
            filtered = reading[zi*sz, ::sy, ::sx]
            # Filter based on zmin
            if zmin:
                # Find bad pixels and make them black
                bad_pixels = np.isin(filtered, good_ids, invert=True)
                filtered[bad_pixels] = 0
            # Write slice to output
            written[zi, :, :] = filtered
            progress('write', 100, out_shape[0], zi)

