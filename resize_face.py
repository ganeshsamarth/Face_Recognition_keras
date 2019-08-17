from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import cv2 

def main(args):
    src_dir = args.input_dir
    dest_dir = args.output_dir
    new_shape = (182, 182)

    if not src_dir.endswith('/'):
        src_dir += '/'
    if not dest_dir.endswith('/'):
        dest_dir += '/'
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for celeb in os.listdir(src_dir):
        celeb_dir = src_dir + celeb + '/'
        dest_celeb_dir = dest_dir + celeb + '/'
        if not os.path.exists(dest_celeb_dir):
            os.makedirs(dest_celeb_dir)
        for img_name in os.listdir(celeb_dir):
            # print("Resizing: ", img_name)
            file_name = celeb_dir + img_name 
            image = cv2.imread(file_name)
            new_image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)

            print(dest_celeb_dir + img_name)
            status = cv2.imwrite(dest_celeb_dir + img_name, new_image)
            if status is True:
                print("")
            else:
                print("0")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', type=str, help='Directory with source images.')
    parser.add_argument('output_dir', type=str, help='Output directory to store resized images')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
        