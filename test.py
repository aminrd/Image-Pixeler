MIT License
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
__author__      = "Amin Aghaee"

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