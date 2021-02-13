import argparse
from scipy import spatial
import numpy as np
from tqdm import tqdm
import cv2
import os


def sim(frame1, frame2):
    flist = []
    for f in [frame1.flatten(), frame2.flatten()]:
        if f.max() == f.min():
            f[np.random.randint(len(f))] += 1
        flist.append(f / 255)

    similarty = 1 - spatial.distance.cosine(flist[0], flist[1])
    return similarty


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video',
                        default='./Portfolio/vancouver.mp4',
                        help='Path to the file or a publicly available link to video'
                        )
    parser.add_argument('-o', '--output',
                        default='./output/',
                        help='Output directory to save frames selected diversely'
                        )
    parser.add_argument('-di', '--diversity',
                        type=float,
                        default=0.25,
                        help='Diversity index showing how diverse the frames should be [0-1] lowest 0 and highest 1. Suggested value = 0.25'
                        )
    args, leftovers = parser.parse_known_args()

    video = cv2.VideoCapture(args.video)
    if not video.isOpened():
        raise FileNotFoundError("File or link could not be found!")

    if not 0 <= args.diversity <= 1:
        raise Exception('Diversity index should be between 0 and 1')

    selected_thumbnail = []

    bar = tqdm(total=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
               desc=f'Reading video frames')

    ret, frame = video.read()
    t_size = (5, int(frame.shape[1] * (5 / frame.shape[0])))
    while ret:
        bar.update()
        frame_thumbnail = cv2.resize(frame, t_size)

        # Check if the new frame is dis-similar to already selected frames
        if frame_thumbnail.max() > 1 and \
            (len(selected_thumbnail) < 1 or\
                all(sim(frame_thumbnail, f) < (1 - args.diversity) for f in selected_thumbnail)):

            print(f"selected! {len(selected_thumbnail)}")

            selected_thumbnail.append(frame_thumbnail)

            file_path = os.path.join(args.output, f"{len(selected_thumbnail)}.jpg")
            cv2.imwrite(file_path, frame)

        ret, frame = video.read()

    video.release()
    print(f'{len(selected_thumbnail)} frames successfully saved in {args.output}')
