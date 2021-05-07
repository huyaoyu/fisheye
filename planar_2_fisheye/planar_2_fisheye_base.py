
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2021-05-06

import numpy as np

class Planar2Fisheye(object):
    def __init__(self, fov, shape, a, s, p):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        shape (array): 2-element. H, W.
        a (array): 4-element, the polynomial coefficents a0, a2, a3, a4.
        s (array): 2-by-2, the strech coefficients.
        p (array): 2-element, the principle point, x, y. 
        '''
        super(Planar2Fisheye, self).__init__()
        
        self.fov = fov # Degree.
        self.shape = shape # H, W.

        # https://www.mathworks.com/help/vision/ug/fisheye-calibration-basics.html#mw_f299f68d-403b-45e0-9de5-55203a09460d
        # and 
        # Scaramuzza, D., A. Martinelli, and R. Siegwart. "A Toolbox for Easy Calibrating Omnidirectional Cameras." Proceedings to IEEE International Conference on Intelligent Robots and Systems, (IROS). Beijing, China, October 7–15, 2006.
        
        # The polynomial coefficents.
        self.a = a # a0, a2, a3, a4.

        self.s = s # The Stretch matrix
        self.invS = np.linalg.inv(self.s)
        self.p = p.reshape((-1, 1)) # The principle point, x, y.

        self.flagCuda = False

    def enable_cuda(self):
        self.flagCuda = True

    def mesh_grid_pixels(self, flagFlatten=False):
        '''Get a mesh grid of the pixel coordinates. '''

        x = np.arange( self.shape[1], dtype=np.int32 )
        y = np.arange( self.shape[0], dtype=np.int32 )

        xx, yy = np.meshgrid(x, y)

        if ( flagFlatten ):
            return xx.reshape((-1)), yy.reshape((-1))
        else:
            return xx, yy

    def pixel_coor_2_distorted(self, pixelCoor):
        '''
        Arguments:
        pixelCoor (array): 2 x n array, the pixel coordinates.

        Returns:
        distCoor (array): 2 x n array, the distorted coordinates.
        '''

        return self.invS @ (pixelCoor - self.p)

    def distorted_2_perspective(self, distortedCoor):
        rho = np.linalg.norm(distortedCoor, axis=0)
        z = self.a[0] + self.a[1] * rho**2 + self.a[2] * rho**3 + self.a[3] * rho**4
        z = z.reshape((1, -1)).astype(np.float32)

        return np.concatenate( ( distortedCoor, z ), axis=0 )

    def get_xyz(self):
        # The pixel coordinates.
        xx, yy = self.mesh_grid_pixels(flagFlatten=True)
        pixelCoor = np.stack( (xx, yy), axis=0 )

        # The distorted coordinates.
        distCoor = self.pixel_coor_2_distorted(pixelCoor).astype(np.float32)

        # The a and b values.
        xyz = self.distorted_2_perspective(distCoor)

        return xyz