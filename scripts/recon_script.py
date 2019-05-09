gpu = 3  # specify which gpu to use
# load blind search zero and recon
import sys
sys.path.insert(0, '/home/heliu/work/dev/v0.2/HEXOMAP/')
import numpy as np
from hexomap import reconstruction  # g-force caller
from hexomap import MicFileTool     # io for reconstruction rst
from hexomap import IntBin          # io for binary image (reduced data)
from hexomap import config
import pycuda.driver as cuda

cuda.init()
device = cuda.Device(gpu) # enter your gpu id here
ctx = device.make_context()

################# configuration #########################
Au_Config={
    'micsize' : np.array([20, 20]),
    'micVoxelSize' : 0.001,
    'micShift' : np.array([0.0, 0.0, 0.0]),
    'expdataNDigit' : 6,
    'energy' : 65.351,      #55.587 # in kev
    'sample' : 'gold',
    'maxQ' : 9,
    'etalimit' : 81 / 180.0 * np.pi,
    'NRot' : 180,
    'NDet' : 2,
    'searchBatchSize' : 6000,
    'reverseRot' : True,          # for aero, is True, for rams: False
    'detL' : np.array([[4.53571404, 6.53571404]]),
    'detJ' : np.array([[1010.79405782, 1027.43844558]]),
    'detK' : np.array([[2015.95118521, 2014.30163539]]),
    'detRot' : np.array([[[89.48560133, 89.53313565, -0.50680978],
  [89.42516322, 89.22570012, -0.45511278]]]),
    'fileBin' : '../examples/johnson_aug18_demo/Au_reduced_1degree//Au_int_1degree_suter_aug18_z',
    'fileBinDigit' : 6,
    'fileBinDetIdx' : np.array([0, 1]),
    'fileBinLayerIdx' : 0,
    '_initialString' : 'demo_gold_'}
    
c = config.Config(**Au_Config)
################# reconstruction #########################
try:
    S.clean_up()
except NameError:
    pass
S = reconstruction.Reconstructor_GPU(ctx=ctx)
S.load_config(c)
S.serial_recon_multi_stage()
################# visualization #########################
#MicFileTool.plot_mic_and_conf(S.squareMicData, 0.5)

################# clean exit ###########################
ctx.pop()