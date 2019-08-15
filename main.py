import Import
import Core
import argparse
import cv2


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input', default='./mainImage.jpg')
    parser.add_argument('-out', '--output', default='output.jpg', help='Name of output file')
    parser.add_argument('-g', '--gallery', default='./MY_IMAGES/')
    parser.add_argument('-s', '--scale', type=int, default=8, help='Output image will be scaled up by this factor [H * $(scale), W * $(scale)]')
    parser.add_argument("--grey", help="Work in Greyscale mode")
    parser.add_argument("-r", "--recursive", help='Scan all images inside Gallery directory recursively')
    args, leftovers = parser.parse_known_args()


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
    window_size = 8
    gallery_im_siz = 128

    gallery = Import.Importer(args.gallery, './output/', recursion=rec, im_size=gallery_im_siz , Gray=grey)

    ncluster = min(len(gallery), (IMAGE.shape[0] * IMAGE.shape[1])//(window_size*window_size))
    C = Core.Core(IMAGE, window_size, n_cluster=ncluster)

    out = C.build(gallery, target_pixel_size=args.scale//window_size)
    cv2.imwrite(args.output, out)
