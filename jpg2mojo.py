#!/usr/bin/python
import os
import cv2
import glob
import argparse
import numpy as np
from toMojo.np2imgo import Imgo
from toMojo.np2sego import Sego

help = {
    'out': 'output mojo parent (default mojo)',
    'jpg2mojo': 'Stack all jpgs into a mojo folder!',
    'jpgs': 'input folder with all jpgs (default jpgs)',
    't': 'datatype for output file (default uint8)',
    'c': '-c enables -t uint32 (and default -o bgr)',
    'o': 'Little Endian channel order as rgba,bgr (default none)',
}
paths = {}
stack = {}
rgba = {
    'r': 0,
    'g': 1,
    'b': 2,
    'a': 3
}
parser = argparse.ArgumentParser(description=help['jpg2mojo'])
parser.add_argument('-t', metavar='string', default='uint8', help=help['t'])
parser.add_argument('-o', metavar='string', default='', help=help['o'])
parser.add_argument('jpgs', default='jpgs', nargs='?', help=help['jpgs'])
parser.add_argument('out', default='mojo', nargs='?', help=help['out'])
parser.add_argument('-c', help=help['c'], action='store_true')

# attain all arguments 
args = vars(parser.parse_args())
for key in ['jpgs', 'out']:
    paths[key] = os.path.realpath(os.path.expanduser(args[key]))
[order, color, dtype] = [args['o'], args['c'], args['t']]

# Set color datatype
if color:
    dtype = 'uint32'
    order = order or 'bgr'
dtype = getattr(np,dtype)

# read all jpgs in jpgs folder
search = os.path.join(paths['jpgs'],'*.jpg')
stack = sorted(glob.glob(search))

# Size input files
sliceShape = cv2.imread(stack[0], 0).shape
shape = (len(stack),) + sliceShape

# Open an output file
outfile = Imgo(paths['out'])
if order:
    outfile = Sego(paths['out'])

# Add each jpg file as a slice
for zi, file in enumerate(stack):
    written = np.zeros(sliceShape,dtype=dtype)
    if not order:
        written = cv2.imread(file, 0).astype(dtype)
    else:
        # pixel to integer
        volume = cv2.imread(file)
        for ci, char in enumerate(order):
            colorbyte = volume[:, :, rgba[char]] * (256 ** ci)
            written = written + colorbyte
    # Write as image or segmentation
    outfile.run(written,zi) 

# Write metadata to ouput file
outfile.save(shape)

