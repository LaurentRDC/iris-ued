# -*- coding: utf-8 -*-
import core
import numpy as n
import PIL.Image

impath = 'D:\\2016.01.06.15.35.TzeroTest_VO2_14mW_Coarse\\processed\\data.timedelay.+40.00.average.pumpon.tif'
im = n.array(PIL.Image.open(impath), n.float)
x,y,r = core.fCenter(1000,1000,200, im)