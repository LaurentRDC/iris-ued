# -*- coding: utf-8 -*-

import numpy as n
import scipy.signal as signal
import matplotlib.pyplot as plt
import PIL.Image as Image
import core as fc


im = n.array(Image.open('C:\Users\Laurent\Dropbox\Powder\VO2\NicVO2\NicVO2_2.tif'), dtype = n.float)
bg = n.array(Image.open('C:\\Users\\Laurent\\Dropbox\\Powder\\VO2\\NicVO2\\bg.tif'), dtype = n.float)
im = im - bg
im[im < 0] = 0
plt.imshow(im)

