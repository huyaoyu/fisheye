
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2021-03-13

import cv2
import math
from numba import (jit, cuda)
import numpy as np

# Local package.
from planar_2_fisheye.planar_2_fisheye_base import Planar2Fisheye

# -->       +---------+
# |  x      |0,W      |0,2W
# v         |    4    |
#  y        |   top   |
# +---------+---------+---------+
# |H,0      |H,W      |H,2W     |H,3W
# |    3    |    0    |    1    |
# |   left  |  front  |  right  |
# +---------+---------+---------+
#  2H,0     |2H,W     |2H,2W     2H,3W
#           |    2    |
#           |  bottom |
#           +---------+
#           3H,W     3H,2W

@jit(nopython=True)
def sample_coor(xyz, 
    offsetX=np.array([1, 2, 1, 0, 1], dtype=np.int32),
    offsetY=np.array([1, 1, 2, 1, 0], dtype=np.int32),
    fov=np.pi):
    m = np.zeros( ( 2, xyz.shape[1] ), dtype=np.float32 )

    oneFourthPi   = np.pi / 4
    halfPi        = np.pi / 2
    threeFourthPi = oneFourthPi + halfPi
    twoPi         = np.pi * 2
    
    halfFov = fov / 2

    for i in range(xyz.shape[1]):
        x = xyz[0, i]
        y = xyz[1, i]
        z = xyz[2, i]

        ax  = np.arctan2( x, z )
        ay  = np.arctan2( y, z )
        axy = np.arctan2( np.sqrt( x**2 + y**2 ), z )

        if ( axy > halfFov ):
            m[0, i] = -1
            m[1, i] = -1
            continue

        t  = np.arctan2( y, x )

        if ( -oneFourthPi <= ax and ax <= oneFourthPi and \
             -oneFourthPi <= ay and ay <= oneFourthPi ):
            # Front.
            m[0, i] = min( max( ( 1 + x/z ) / 2, 0 ), 1 ) + offsetX[0]
            m[1, i] = min( max( ( 1 + y/z ) / 2, 0 ), 1 ) + offsetY[0]
        elif ( -oneFourthPi <= t and t < oneFourthPi ):
            # Right.
            m[0, i] = min( max( ( 1 - z/x ) / 2, 0 ), 1 ) + offsetX[1]
            m[1, i] = min( max( ( 1 + y/x ) / 2, 0 ), 1 ) + offsetY[1]
        elif ( threeFourthPi <= t or t < -threeFourthPi ):
            # Left.
            m[0, i] = min( max( ( 1 - z/x ) / 2, 0 ), 1 ) + offsetX[3]
            m[1, i] = min( max( ( 1 - y/x ) / 2, 0 ), 1 ) + offsetY[3]
        elif ( oneFourthPi <= t and t < threeFourthPi ):
            # Bottom.
            m[0, i] = min( max( ( 1 + x/y ) / 2, 0 ), 1 ) + offsetX[2]
            m[1, i] = min( max( ( 1 - z/y ) / 2, 0 ), 1 ) + offsetY[2]
        elif ( -threeFourthPi <= t and t < -oneFourthPi ) :
            # Top
            m[0, i] = min( max( ( 1 - x/y ) / 2, 0 ), 1 ) + offsetX[4]
            m[1, i] = min( max( ( 1 - z/y ) / 2, 0 ), 1 ) + offsetY[4]
        else:
            raise Exception('xy invalid.')
            # raise Exception(f'x = {x}, y = {y}, z = {z}, ax = {ax}, ay = {ay}, oneFourthPi = {oneFourthPi}, halfPi = {halfPi}')

    return m

@cuda.jit()
def k_sample_coor(
    output, xyz, offsets, fov):
    # Prepare the index.
    xIdx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    xStride = cuda.blockDim.x * cuda.gridDim.x

    # Constants.
    oneFourthPi   = np.pi / 4
    halfPi        = np.pi / 2
    threeFourthPi = oneFourthPi + halfPi
    twoPi         = np.pi * 2
    halfFov       = fov / 2

    # Loop.
    for i in range( xIdx, xyz.shape[1], xStride ):
        x = xyz[0, i]
        y = xyz[1, i]
        z = xyz[2, i]

        ax  = math.atan2( x, z )
        ay  = math.atan2( y, z )
        axy = math.atan2( math.sqrt( x**2 + y**2 ), z )

        if ( axy > halfFov ):
            output[0, i] = -1
            output[1, i] = -1
            continue

        t = math.atan2( y, x )

        if ( -oneFourthPi <= ax and ax <= oneFourthPi and \
             -oneFourthPi <= ay and ay <= oneFourthPi ):
            # Front.
            output[0, i] = min( max( ( 1 + x/z ) / 2, 0 ), 1 ) + offsets[0]
            output[1, i] = min( max( ( 1 + y/z ) / 2, 0 ), 1 ) + offsets[5]
        elif ( -oneFourthPi <= t and t < oneFourthPi ):
            # Right.
            output[0, i] = min( max( ( 1 - z/x ) / 2, 0 ), 1 ) + offsets[1]
            output[1, i] = min( max( ( 1 + y/x ) / 2, 0 ), 1 ) + offsets[6]
        elif ( threeFourthPi <= t or t < -threeFourthPi ):
            # Left.
            output[0, i] = min( max( ( 1 - z/x ) / 2, 0 ), 1 ) + offsets[3]
            output[1, i] = min( max( ( 1 - y/x ) / 2, 0 ), 1 ) + offsets[8]
        elif ( oneFourthPi <= t and t < threeFourthPi ):
            # Bottom.
            output[0, i] = min( max( ( 1 + x/y ) / 2, 0 ), 1 ) + offsets[2]
            output[1, i] = min( max( ( 1 - z/y ) / 2, 0 ), 1 ) + offsets[7]
        elif ( -threeFourthPi <= t and t < -oneFourthPi ) :
            # Top
            output[0, i] = min( max( ( 1 - x/y ) / 2, 0 ), 1 ) + offsets[4]
            output[1, i] = min( max( ( 1 - z/y ) / 2, 0 ), 1 ) + offsets[9]

class FivePlanar2Fisheye(Planar2Fisheye):
    def __init__(self, fov, shape, a, s, p):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        shape (array): 2-element. H, W.
        a (array): 4-element, the polynomial coefficents a0, a2, a3, a4.
        s (array): 2-by-2, the strech coefficients.
        p (array): 2-element, the principle point, x, y. 
        '''
        super(FivePlanar2Fisheye, self).__init__(fov, shape, a, s, p)

    def make_image_cross(self, imgs):
        '''
        Arguments:
        imgs (list of arrays): The five images in the order of front, right, bottom, left, and top.

        Returns:
        A image cross with shape (3*H, 3*W).
        '''
        H, W = imgs[0].shape[:2]

        if ( imgs[0].ndim == 3 ):
            canvas = np.zeros( ( 3*H, 3*W, 3 ), dtype=np.uint8 )
        elif ( imgs[0].ndim == 2 ):
            canvas = np.zeros( ( 3*H, 3*W ), dtype=np.uint8 )
        else:
            raise Exception(f'Wrong dimension of the input images. imgs[0].shape = {imgs[0].shape}')

        canvas[  H:2*H,   W:2*W, ...] = imgs[0] # Front.
        canvas[  H:2*H, 2*W:3*W, ...] = imgs[1] # Right.
        canvas[2*H:3*H,   W:2*W, ...] = imgs[2] # Bottom.
        canvas[  H:2*H,   0:W,   ...] = imgs[3] # Left.
        canvas[  0:H,     W:2*W, ...] = imgs[4] # Top.

        # Padding.
        # Left.
        canvas[ H-1,   2*W:3*W, ... ] = imgs[4][ ::-1, -1, ... ]
        canvas[ 2*H, 2*W:3*W, ... ] = imgs[2][  :,   -1, ... ]
        # Bottom.
        canvas[ 2*H:3*H, 2*W, ... ] = imgs[1][ -1, :, ... ]
        canvas[ 2*H:3*H, W-1, ... ] = imgs[3][ -1, ::-1, ... ]
        # Left.
        canvas[ H-1, 0:W, ... ] = imgs[4][:, 0, ...]
        canvas[ 2*H, 0:W, ... ] = imgs[2][::-1, 0, ...]
        # Top.
        canvas[ 0:H, W-1, ... ] = imgs[3][0, :, ...]
        canvas[ 0:H, 2*W, ... ] = imgs[1][0, ::-1, ...]

        return canvas

    def sample_coor_cuda(self, 
        xyz, 
        offsetX=np.array([1, 2, 1, 0, 1], dtype=np.int32),
        offsetY=np.array([1, 1, 2, 1, 0], dtype=np.int32),
        fov=np.pi):

        # Prepare the memory.
        dXyz     = cuda.to_device(xyz)
        # dOffsets = cuda.to_device( 
        #     np.concatenate( (offsetX, offsetY) ) )
        output  = np.zeros_like(xyz)
        dOutput = cuda.to_device(output)

        cuda.synchronize()
        k_sample_coor[[1024,1,1],[256,1,1]](
            dOutput, dXyz, np.concatenate( (offsetX, offsetY) ), fov)
        cuda.synchronize()

        output = dOutput.copy_to_host()
        print(f'output.dtype = {output.dtype}')

        return output

    def __repr__(self):
        s = f'''fov = {self.fov}
shape = {self.shape}
a = {self.a}
s = 
{self.s}
invs = 
{self.invS}
p = {self.p}
flagCuda = {self.flagCuda}
'''
        return s

    def __call__(self, imgs):
        '''
        Arguments:
        imgs (list of arrays): The five images in the order of front, right, bottom, left, and top.

        Returns:
        The generated fisheye image.
        '''

        assert( len(imgs) == 5 ), f'len(imgs) = {len(imgs)}'

        # Make the image cross.
        imgCross = self.make_image_cross( imgs )

        # Get the original shape of the input images.
        H = imgCross.shape[0] // 3
        W = imgCross.shape[1] // 3

        # print(f'imgCross.shape = {imgCross.shape}')
        # print(f'W = {W}')

        # The 3D coordinates of the hyper-surface.
        xyz = self.get_xyz()

        # Sample.
        if ( self.flagCuda ):
            m = self.sample_coor_cuda( xyz, fov=self.fov/180*np.pi )
        else:
            m = sample_coor(xyz, fov=self.fov/180*np.pi)

        negativeX = m[0, :] < 0
        negativeY = m[1, :] < 0
        negative  = np.logical_and( negativeX, negativeY )
        negative  = negative.reshape(self.shape)

        # print(f'm.max() = {m.max()}')
        # print(f'm.min() = {m.min()}')

        mx = m[0, :].reshape(self.shape)
        my = m[1, :].reshape(self.shape)

        # Sample.
        sampled = cv2.remap( 
            imgCross, 
            mx * (W-1), my * (H-1), 
            interpolation=cv2.INTER_LINEAR )

        # Apply gray color on negative coordinates.
        sampled[negative, :] = 127

        return sampled

        