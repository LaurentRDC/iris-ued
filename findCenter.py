# -*- coding: utf-8 -*-
"""
0000000000000000000000000000000000000000000000000000000000000000000000000000000
findCenter

0000000000000000000000000000000000000000000000000000000000000000000000000000000
"""

#Basics
import numpy as n
import scipy.optimize as opt

#plotting
import matplotlib.pyplot as plt

# For importing TIFF images
from PIL import Image

def fCenter(xg, yg, rg, im):
    """
    Finds the center of a diffraction pattern based on an initial guess of the center.
    
    Parameters
    ----------
    xg, yg, rg : ints
        Guesses for the (x,y) position of the center, and the radius
    im : ndarray, shape (N,N)
        ndarray of a TIFF image
    
    Returns
    -------
    optimized center and peak position
    
    See also
    --------
    Scipy.optimize.fmin - Minimize a function using the downhill simplex algorithm
    """

    #find maximum intensity
    c = lambda x: circ(x[0],x[1],x[2],im)

    return opt.minimize(c,[xg,yg,rg]).x

def circ(xg,yg,rg,im):
    """
    Sums the intensity over a circle of given radius and center position
    on an image.
    
    Parameters
    ----------
    xg, yg, rg : ints
        The (x,y) position of the center, and the radius
    im : ndarray, shape (N,N)
        ndarray of a TIFF image
    
    Returns
    -------
    Total intensity at pixels on the given circle. 
    
    """
     #image size
    s = im.shape[0]
    
    xMat, yMat = n.meshgrid(n.linspace(1, s, s),n.linspace(1, s, s))
    # find coords on circle and sum intensity
    xvals, yvals = n.where(((n.around(n.sqrt((xMat-xg)**2+(yMat-yg)**2))-n.around(rg)) < .1) & (yMat > 550))
    ftemp = n.sum(im[xvals, yvals])
    print xg
    print yg
    print ftemp
    
    return 1/ftemp


#Manual guesses
#xg = 560
#yg = 540
#rg = n.sqrt((468-xg)**2 + (543-yg)**2)
#
##import image
##filename = 'C:\Users\SiwickWS1\Dropbox\Powder\VO2\NicVO2\NicVO2_2.tif'#temp glob var
##im = n.array(Image.open(filename))#temp glob var
#
##Plot image
#fig = plt.figure(1)
#fig.clf()
#ax = fig.add_subplot(111)
#ax.imshow(im)
#
##Find center of the image
#x = fCenter(xg, yg, rg, im)
#
##Create a circle to overlay on the image and plot
#xval = x[0]+ x[2]*n.cos(n.linspace(0,2*n.pi,100))
#yval = x[1]+ x[2]*n.sin(n.linspace(0,2*n.pi,100))
#ax.scatter(xval,yval)
#fig.show()
