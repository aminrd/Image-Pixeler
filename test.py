import numpy as np
import Core


# Image is 128x128
IMG = np.random.random((128, 128, 3))

# Considering every 8x8 patch as one => 16x16 patch
C = Core.Core(IMG, 8, n_cluster = 16)

# Images in Gallery are all 512x512 pixels
gallery = np.random.random((500, 512, 512, 3))

# Converting every patch to 64x64 size
b = C.build(gallery, target_pixel_size=64)

Image.fromarray(np.uint8(b * 255)).show()