import numpy as np
import cv2
from sklearn.cluster import KMeans
from ..Import.Importer import pixel_image_error

def img_to_features(img, patch_size=4):    
    ps = patch_size
    H = img.shape[0] // ps
    W = img.shape[1] // ps
    
    if len(img) == 2:
        patches = np.zeros((1, ps, ps))
    else:
        patches = np.zeros((1, ps, ps, 3))
            
    for h in range(H):
        for w in range(W):
            patch = img[h*ps: (h+1)*ps, w*ps: (w+1)*ps, ...]
            patch = np.expand_dims(patch, 0)
            print(patches.shape)
            print(patch.shape)
            patches = np.concatenate((patches, patch),0)
            
    return patches[1:]

class Core:
    def __init__(self, img, window_size=4 ,n_cluster=26):
        self.img = img
        
        if self.img.shape[0] % window_size != 0:
            imax = self.img.shape[0] - self.img.shape[0] % window_size
            self.img = self.img[ :imax, ...]
            
        if self.img.shape[1] % window_size != 0:
            imax = self.img.shape[1] - self.img.shape[1] % window_size
            self.img = self.img[ :, :imax, ...]

        X = img_to_features(img, window_size)
        self.kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)

    def find_best_matches(self, gallery):
        G = []
        for i in range(len(self.kmeans.cluster_centers_)):
            best_img = gallery[0]
            emin = np.Inf
            for im in gallery:
                E = pixel_image_error(im, self.kmeans.cluster_centers_[i])
                if E < emin:
                    emin = E
                    best_img = im
            G += [best_img]
            
        return G

