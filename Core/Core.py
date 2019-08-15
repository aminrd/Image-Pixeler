import numpy as np
import cv2
from sklearn.cluster import KMeans

def pixel_image_error(img, pixarr):
    pixarr = np.array(pixarr)
    image = cv2.resize(img, (pixarr.shape[0],pixarr.shape[1]))
    return np.sum( np.square(pixarr - image)) // (pixarr.shape[0] * pixarr.shape[1])

def img_to_features(img, patch_size=4):    
    ps = patch_size
    H = img.shape[0] // ps
    W = img.shape[1] // ps
    
    if len(img) == 2:
        patches = np.zeros((1, ps*ps))
    else:
        patches = np.zeros((1, ps*ps*3))
            
    for h in range(H):
        for w in range(W):
            patch = img[h*ps: (h+1)*ps, w*ps: (w+1)*ps, ...]
            patch = np.expand_dims(patch, 0)
            patch = np.reshape(patch, (1, np.size(patch)))
            patches = np.concatenate((patches, patch),0)
            
    return patches[1:]

class Core:
    def __init__(self, img, window_size=4 ,n_cluster=16):
        self.img = img
        self.window_size = window_size
        
        if self.img.shape[0] % window_size != 0:
            imax = self.img.shape[0] - self.img.shape[0] % window_size
            self.img = self.img[ :imax, ...]
            
        if self.img.shape[1] % window_size != 0:
            imax = self.img.shape[1] - self.img.shape[1] % window_size
            self.img = self.img[ :, :imax, ...]

        X = img_to_features(img, window_size)
        self.kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)

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
        return H,W

    def build(self, gallery, target_pixel_size=64):
        G, esum = self.find_best_matches(gallery, target_pixel_size)

        Hp = (self.img.shape[0] // self.window_size)
        Wp = (self.img.shape[1] // self.window_size)
        H = Hp * target_pixel_size
        W = Wp * target_pixel_size

        if len(self.img.shape) > 2:
            art = np.zeros((H,W,3))
        else:
            art = np.zeros((H, W))

        for l in range(len(self.kmeans.labels_)):
            L = self.kmeans.labels_[l]
            Gimg = G[L]
            
            # Find proper indices: 
            i = l // Hp
            j = l % Wp

            I = i * target_pixel_size
            J = j * target_pixel_size

            art[I:I+target_pixel_size, J:J+target_pixel_size, ...] = Gimg

        return art, esum
    