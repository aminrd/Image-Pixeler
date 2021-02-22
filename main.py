# MIT License
#
# Copyright (c) 2019 Amin Aghaee
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
__author__ = "Amin Aghaee"

import Import
import Core
import argparse
import cv2
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input', default='./mainImage.jpg')
    parser.add_argument('-out', '--output', default='output.jpg', help='Name of output file')
    parser.add_argument('-g', '--gallery', default='./Gallery/')
    parser.add_argument('-s', '--scale', type=int, default=8,
                        help='Output image will be scaled up by this factor [H * $(scale), W * $(scale)]')
    parser.add_argument("--grey", help="Work in Greyscale mode")
    parser.add_argument("-r", "--recursive", help='Scan all images inside Gallery directory recursively')
    parser.add_argument("-v", "--verbose", help='Print steps in system log')
    args, leftovers = parser.parse_known_args()

    if args.verbose is not None:
        verbose = True
    else:
        verbose = False

    if args.grey is not None:
        grey = True
    else:
        grey = False

    if args.recursive is not None:
        rec = True
    else:
        rec = False

    if grey:
        IMAGE = cv2.imread(args.input, 0)
    else:
        IMAGE = cv2.imread(args.input, 1)

        # Play with these arguments:
    window_size = 4
    gallery_im_siz = 32

    gallery = Import.Importer(args.gallery, './output/', recursion=rec, im_size=gallery_im_siz, Gray=grey,
                              verbose=verbose)

    ws = window_size
    ncluster = min(len(gallery), (IMAGE.shape[0] * IMAGE.shape[1]) // ((ws + 1) ** 2))
    C = Core.Core(IMAGE, window_size, n_cluster=ncluster)

    # out, esum = C.build(gallery, target_pixel_size=args.scale // window_size)
    out = C.build_cosine(gallery)
    esum = 0
    out = np.uint8(out)
    print('Error = {}'.format(esum))
    cv2.imwrite(args.output, out)
