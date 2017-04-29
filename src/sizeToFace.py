#!/usr/bin/python
# sizeToFace.py: face detection using OpenCV

import FaceImage
from multiprocessing import Pool
import sys
import os


def main():
    # Print usage if no args specified
    args = sys.argv[1:]
    if len(args) != 2:
        print(
            'Usage: python sizeToFace.py <image_directory> <output_directory>')
        return

    # Get input files, sort by last modified time
    in_dir = os.path.abspath(args[0])
    out_dir = os.path.abspath(args[1])
    files = []
    for root, dirs, files in os.walk(in_dir):
        jpegs = [
            os.path.join(root, f) for f in files if f.lower().endswith('.jpg')
        ]
        if not jpegs:
            continue
        relpath = os.path.relpath(root, in_dir)
        to_add = [os.path.join(relpath, f) for f in jpegs]
        files.extend(to_add)

    if len(files) == 0:
        print('No jpg files found in ' + args[0])
        return

    pool = Pool()

    # For every JPG in the given directory
    for file_path in files:
        out_path = os.path.join(out_dir, file_path)
        in_path = os.path.join(in_dir, file_path)

        print('Added to pool: ' + in_path + ' with output path: ' + out_path)
        pool.apply_async(FaceImage.runFaceImage, (in_path, out_path))

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
