'''
This is a file for testing purpose
Usage:
    cd to module root directory
    run: python -m hexomap
'''
from hexomap import config
import numpy as np
from hexomap import reconstruction
from hexomap import MicFileTool
import os
import hexomap

c = config.Config().load(os.path.abspath(os.path.join(hexomap.__file__ ,"../..")) +'/examples/johnson_aug18_demo/demo_gold_twiddle_3.h5')
print(c)
c.fileBin = os.path.abspath(os.path.join(hexomap.__file__ ,"../..")) +'/examples/johnson_aug18_demo/Au_reduced_1degree/Au_int_1degree_suter_aug18_z'
c.micVoxelSize = 0.005
c.micsize = [15, 15]

print(c)

try:
    S.clean_up()
except NameError:
    pass

S = reconstruction.Reconstructor_GPU()
S.load_config(c)
S.serial_recon_multi_stage()

#MicFileTool.plot_mic_and_conf(S.squareMicData, 0.5)
