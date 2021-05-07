
# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>
# Date:
# 2021-05-06

import cv2
import numpy as np

# Local package.
from planar_2_fisheye.planar_2_fisheye_base import Planar2Fisheye

# Reference frames are defined in
# https://docs.google.com/presentation/d/1xfGt3oHgjvG_eGtKMlDg8MTYJhARNVKcTMcPl3s118o/edit?usp=sharing

class FullView2Fisheye(Planar2Fisheye):
    def __init__(self, fov, shape, a, s, p):
        '''
        Arguments:
        fov (float): Full FoV of the lens in degrees.
        shape (array): 2-element. H, W.
        a (array): 4-element, the polynomial coefficents a0, a2, a3, a4.
        s (array): 2-by-2, the strech coefficients.
        p (array): 2-element, the principle point, x, y. 
        '''
        super(FullView2Fisheye, self).__init__(fov, shape, a, s, p)

    def get_lon_lat(self):
        # Get the xyz 3D coordinates of the 3D surface.
        xyz = self.get_xyz()

        # The distance from xyz point to the center.
        d = np.linalg.norm( xyz, axis=0 )

        # Longitude and latitude.
        lon_lat = np.zeros( (2, xyz.shape[1]), dtype=np.float32 )
        lon_lat[0, :] = np.pi - np.arctan2( xyz[2, :], xyz[0, :] )
        lon_lat[1, :] = np.pi - np.arccos( xyz[1, :] / d )

        # Handle FoV.
        z_angle = np.arccos( xyz[2, :] / d )
        out_of_fov = z_angle > self.fov / 2 / 180.0 * np.pi

        return lon_lat, out_of_fov

    def __call__(self, img, lon_shift=np.pi):
        # Get the shape of the input image.
        H, W = img.shape[:2]

        # Get the longitude and latitude coordinates.
        lon_lat, out_of_fov = self.get_lon_lat()

        # Shift the longitude values.
        lon_lat[0, :] += lon_shift
        rewind_mask = lon_lat[0, :] >= 2 * np.pi
        lon_lat[0, rewind_mask] -= 2 * np.pi

        # Get the sample location.
        sx = ( lon_lat[0, :] / ( 2 * np.pi ) * (W-1) ).reshape(self.shape)
        sy = ( lon_lat[1, :] / np.pi * (H-1) ).reshape(self.shape)

        # Sample.
        sampled = cv2.remap(img, sx, sy, interpolation=cv2.INTER_LINEAR)

        # FoV.
        sampled[ out_of_fov.reshape(self.shape), : ] = 127

        return sampled