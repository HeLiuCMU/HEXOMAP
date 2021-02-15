#!python

import sys
sys.path.insert(0, '..')
import hexomap
from hexomap import reduction

startIdx = 333224
tiffInitial = '/media/heliu/Seagate Backup Plus Drive/krause_jul19/nf/s1350_100_1_nf/s1350_100_1_nf_'
digit = 6
extention = '.tif'
NInt = 4  # integrate 4 images into 1.
NImage = 4 # 1440*22 # number of images before integration
outInitial = '/media/heliu/Seagate Backup Plus Drive/krause_jul19/nf/s1350_100_1_nf/s1350_100_1_nf_int4_'
outStartIdx = 0 # starting index of output image

reduction.integrate_tiff(tiffInitial, startIdx, digit, extention, NImage, NInt,outInitial, outStartIdx)