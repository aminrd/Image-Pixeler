# MIT License
#
# Copyright (c) 2021 Amin Aghaee
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

import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy import spatial
from tqdm import tqdm
import random

def similarty_cosine(frame1, frame2):
    frame1 = np.copy(frame1)
    frame2 = np.copy(frame2)

    flist = []
    for f in [frame1.flatten(), frame2.flatten()]:
        if f.max() == f.min():
            f[np.random.randint(len(f))] += 1
        flist.append(f / 255)

    similarty = 1 - spatial.distance.cosine(flist[0], flist[1])
    return similarty


def pixel_image_error(img, pixarr):
    pixarr = np.array(pixarr)
    image = cv2.resize(img, (pixarr.shape[0], pixarr.shape[1]))
    return np.sum(np.square(pixarr - image)) // (pixarr.shape[0] * pixarr.shape[1])


def img_to_features(img, patch_size=4):
    ps = patch_size
    H = img.shape[0] // ps
    W = img.shape[1] // ps

    if len(img) == 2:
        patches = np.zeros((1, ps * ps))
    else:
        patches = np.zeros((1, ps * ps * 3))

    for h in range(H):
        for w in range(W):
            patch = img[h * ps: (h + 1) * ps, w * ps: (w + 1) * ps, ...]
            patch = np.expand_dims(patch, 0)
            patch = np.reshape(patch, (1, np.size(patch)))
            patches = np.concatenate((patches, patch), 0)

    return patches[1:]


class Core:
    def __init__(self, img, window_size=4, n_cluster=16, verbose=False):
        self.verbose = verbose
        self.img = img
        self.window_size = window_size

        if self.img.shape[0] % window_size != 0:
            imax = self.img.shape[0] - self.img.shape[0] % window_size
            self.img = self.img[:imax, ...]

        if self.img.shape[1] % window_size != 0:
            imax = self.img.shape[1] - self.img.shape[1] % window_size
            self.img = self.img[:, :imax, ...]

        X = img_to_features(img, window_size)

        if self.verbose:
            print('Start training K-Means model')

        self.kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)

        if self.verbose:
            print('Finish training K-Means model, number of clusters : {}'.format(len(self.kmeans.cluster_centers_)))

    def find_best_matches(self, gallery, target_size=4):
        G = []
        error_sum = 0

        for i in range(len(self.kmeans.cluster_centers_)):
            best_img = gallery[0]
            emin = np.Inf
            for im in gallery:

                if len(self.img.shape) > 2:
                    converted = np.reshape(self.kmeans.cluster_centers_[i], (self.window_size, self.window_size, 3))
                else:
                    converted = np.reshape(self.kmeans.cluster_centers_[i], (self.window_size, self.window_size))

                E = pixel_image_error(im, converted)

                if E < emin:
                    emin = E
                    best_img = im

            best_img = cv2.resize(best_img, (target_size, target_size))
            G += [best_img]
            error_sum += emin
        return G, error_sum

    def estimate_size(self, target_pixel_size):
        Hp = (self.img.shape[0] // self.window_size)
        Wp = (self.img.shape[1] // self.window_size)
        H = Hp * target_pixel_size
        W = Wp * target_pixel_size
        return H, W

    def build(self, gallery, target_pixel_size=64):
        G, esum = self.find_best_matches(gallery, target_pixel_size)

        Hp = (self.img.shape[0] // self.window_size)
        Wp = (self.img.shape[1] // self.window_size)
        H = Hp * target_pixel_size
        W = Wp * target_pixel_size

        if len(self.img.shape) > 2:
            art = np.zeros((H, W, 3))
        else:
            art = np.zeros((H, W))

        print('number of labels: ')
        print(len(self.kmeans.labels_))

        if self.verbose:
            print('Start merging pixels to build the art!')

        for l in range(len(self.kmeans.labels_)):

            if self.verbose and np.random.random() < 0.05:
                print('{} out of {}'.format(l, len(self.kmeans.labels_)))

            L = self.kmeans.labels_[l]
            Gimg = G[L]

            # Find proper indices: 
            i = l // Wp
            j = l % Wp

            I = i * target_pixel_size
            J = j * target_pixel_size

            art[I:I + target_pixel_size, J:J + target_pixel_size, ...] = Gimg

        return art, esum

    def build_cosine(self, gallery, scale=4):
        ws = 16
        gallery = list(cv2.resize(b, (ws, ws)) for b in gallery)
        gallery_small = list(cv2.resize(b, (8, 8)) for b in gallery)

        h, w, d = self.img.shape
        art = cv2.resize(self.img, (scale * w, scale * h))

        h, w, d = art.shape
        h = (h // ws) * ws
        w = (w // ws) * ws
        art = art[:h, :w, ...]

        for i in tqdm(range(0, w, ws)):
            for j in range(0, h, ws):
                window = art[j: j+ws, i: i+ws]
                window = cv2.resize(window, (8, 8))

                similarty_vector = list(similarty_cosine(window, g) for g in gallery_small)
                indices = np.argsort(similarty_vector)
                top_2 = indices[-2:]
                chosen = random.choice(top_2)

                art[j: j + ws, i: i + ws, ...] = gallery[chosen]
        return art
