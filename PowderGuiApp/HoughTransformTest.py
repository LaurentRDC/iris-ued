# -*- coding: utf-8 -*-
import numpy as n
import PIL.Image
import matplotlib.pyplot as plt

impath = 'D:\\2016.01.06.15.35.TzeroTest_VO2_14mW_Coarse\\processed\\data.timedelay.+40.00.average.pumpon.tif'
im = n.array(PIL.Image.open(impath), dtype = n.uint16)

def _findCenterHoughTransform(im, guess_center = list()):
    """ This function returns the center as calculated by the Hough Transform """
    import cv2
    from scipy.misc import imresize
    dimension = im.shape[0]        
    if dimension != 2048:
        raise NotImplemented
    
    scales = [16, 8, 4, 2]
    resolutions = [(dimension/scale, dimension/scale) for scale in scales]
    
    #Resize image and find center for multiple resolutions, then compare results
    circles = list()
    for resolution, scale in zip(resolutions, scales):
        resized_im = imresize(im, size = resolution)
        resized_im = (resized_im).astype(n.uint8)
        
        min_radius = int(resolution[0]/20.0)
        max_radius = int(resolution[0]/1.0)
        foo = cv2.HoughCircles(resized_im, cv2.cv.CV_HOUGH_GRADIENT, 2, minDist = 1, minRadius = min_radius, maxRadius = max_radius)
        
        if foo is not None:
            #format centers appropriately
            centers = list()
            
            circles.append(foo)
            
    return circles

circles = _findCenterHoughTransform(im)