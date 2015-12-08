# -*- coding: utf-8 -*-
"""
0000000000000000000000000000000000000000000000000000000000000000000000000000000
findCenter

0000000000000000000000000000000000000000000000000000000000000000000000000000000
"""

import numpy as n
import scipy.optimize

#plotting
import matplotlib.pyplot as plt
import msvcrt as m

# io
from PIL import Image






#0000000000000000000000000000000000000000000000000000000000000000000000000000000
#

#'''
#ARGS - x,y and r guesses, and image intensity map
#RETRUN - optimized center and peak position
def fCenter(xg, yg, rg, im):
    """
    Finds the center of a diffraction pattern based on an initial guess of the center.
    
    Parameters
    ----------
    xg, yg, rg - ints
        Guesses for the (x,y) position of the center, and the radius
    im - ndarray, shape (N,N)
        ndarray of a TIFF image
    
    Returns
    -------
    optimized center and peak position
    """
  


   
   
    
    #find maximum intensity
    c = lambda x:circ(x[0],x[1],x[2],im)
 
    value =  scipy.optimize.fmin(c,[xg,yg,rg])
    
    return value
#'''
#0000000000000000000000000000000000000000000000000000000000000000000000000000000
#
#'''
#ARGS - x,y and r guesses, and image intensity map
#RETURNS - sum of intensity in circle defined by ARGS
def circ(xg,yg,rg,im):
    
     #image size
    s = im.shape[0]
    
    Xmat,Ymat = n.meshgrid(n.linspace(1,s,s),n.linspace(1,s,s))
    # find coords on circle and sum intensity
    vals = n.where((n.around(n.sqrt((Xmat-xg)**2+(Ymat-yg)**2))-n.around(rg))<.1)
    ftemp = n.sum(im[vals])
    print ftemp

    value = 1/ftemp
    return value



    
#'''
#0000000000000000000000000000000000000000000000000000000000000000000000000000000
#
#'''   

    

xg = 560
yg = 540
rg = n.sqrt((468-xg)**2 + (543-yg)**2)




filename = 'C:\Users\SiwickWS1\Dropbox\Powder\VO2\NicVO2\NicVO2_2.tif'#temp glob var
im = n.array(Image.open(filename))#temp glob var


fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(111)
ax.imshow(im)


x = fCenter(xg, yg, rg, im)

xval = x[0]+ x[2]*n.cos(n.linspace(0,2*n.pi,100))
yval = x[1]+ x[2]*n.sin(n.linspace(0,2*n.pi,100))
ax.scatter(xval,yval)





#    
#'''
#def definecircle(guess,img,Xmat,Ymat)
# 
#   rguess = guess[3]
#   xcguess = guess[1]
#   ycguess = guess[2]
#   
#  vals = find(round(sqrt((Xmat-xcguess).^2+(Ymat-ycguess).^2))==round(rguess)) 
#   ftemp = sum(img(vals))
#   F=1/ftemp
#
#   value = F    
#    
#return value
#'''




#
#    '''
#Matlab Version
#
#    
#low = 0;
#high = max(img(:));
#%close all;    
#figure(1);
#% Create start array
#
#
#%imagesc(img,[0 (high-low)/contrast+low]);
#imagesc(img,contrast);
#disp('Click on center');
#[xcguess,ycguess] = ginput(1);
#disp('Click on first order peak');
#[xguess,yguess] = ginput(1);
#
#s = size(img);
#rguess = sqrt((xguess-xcguess)^2+(yguess-ycguess)^2);
#[Xmat,Ymat] = meshgrid(1:1024,1:1024);
#
#
#tic
#fit = fitcircletodiff(xcguess,ycguess,rguess,img,Xmat,Ymat); % find center of pumped image
#toc
#xc = fit(1);
#yc = fit(2);
#r = fit(3);
#vals = find(round(sqrt((Xmat-xc).^2+(Ymat-yc).^2))==round(r)); 
#xarray = Xmat(vals);
#yarray = Ymat(vals);
#%imagesc(img,[0 (high-low)/contrast+low]);
#imagesc(img,contrast);
#hold on;
#
#plot(xarray,yarray,'ko','Markersize',1);
#    '''
#
#    
#    


