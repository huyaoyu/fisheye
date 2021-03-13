
# Author: Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date: 2021-03-13

import cv2
import numpy as np
import time

import os
import sys

# Prepare the Python environment.
import os
_CF       = os.path.realpath(__file__)
_PKG_PATH = os.path.dirname(os.path.dirname(os.path.dirname(_CF)))

import sys
sys.path.insert(0, _PKG_PATH)

from planar_2_fisheye.five_images import FivePlanar2Fisheye

def read_image(fn):
    assert( os.path.isfile(fn) ), \
        f'{fn} does not exist. '

    return cv2.imread(fn, cv2.IMREAD_UNCHANGED)

if __name__ == '__main__':
    print('Hello, %s! ' % ( os.path.basename(__file__) ))

    # Load the images.
    imgFFn = '/home/yaoyu/Playground/Fisheye/pics2pano/test/0_front.png'
    imgRFn = '/home/yaoyu/Playground/Fisheye/pics2pano/test/0_right.png'
    imgBFn = '/home/yaoyu/Playground/Fisheye/pics2pano/test/0_down.png'
    imgLFn = '/home/yaoyu/Playground/Fisheye/pics2pano/test/0_left.png'
    imgTFn = '/home/yaoyu/Playground/Fisheye/pics2pano/test/0_up.png'
    imgFns = [ imgFFn, imgRFn, imgBFn, imgLFn, imgTFn ]
    # Make a list.
    imgs = [ read_image(fn) for fn in imgFns ]

    sampler = FivePlanar2Fisheye(
        fov=195,
        shape=(2100, 2450), 
        a=np.array([585.84, -0.00060634, 1.9243e-07, -3.5623e-10], dtype=np.float32),
        s=np.eye(2,dtype=np.float32), 
        p=np.array([1225, 1050], dtype=np.float32))
    sampler.enable_cuda()

    print(sampler)
    
    cv2.imwrite('ImageCross.png', sampler.make_image_cross(imgs))

    startTime = time.time()
    sampled = sampler(imgs)
    endTime = time.time()

    print(sampled.shape)
    print(f'Sample in {endTime-startTime}s')

    cv2.imwrite('Sampled.png', sampled)
