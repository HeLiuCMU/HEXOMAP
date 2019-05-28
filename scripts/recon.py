#!python
'''
usage: recon.py [-h] [-c CONFIG] [-g GPU]

single thread hedm reconstruction

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        config file, .yml ,.yaml, h5, hdf5
  -g GPU, --gpu GPU     which gpu to use(0-4)
'''
import sys
sys.path.insert(0, '/home/heliu/work/dev/v0.2/HEXOMAP/')
import numpy as np
import os
import hexomap
from hexomap import reconstruction  # g-force caller
from hexomap import MicFileTool     # io for reconstruction rst
from hexomap import IntBin          # io for binary image (reduced data)
from hexomap import config
import argparse
################# configuration #########################
Au_Config={
    'micsize' : np.array([20, 20]),
    'micVoxelSize' : 0.01,
    'micShift' : np.array([0.0, 0.0, 0.0]),
    'micMask' : None,
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
    'fileBin' : os.path.abspath(os.path.join(hexomap.__file__ ,"../..")) + '/examples/johnson_aug18_demo/Au_reduced_1degree/Au_int_1degree_suter_aug18_z',
    'fileBinDigit' : 6,
    'fileBinDetIdx' : np.array([0, 1]),
    'fileBinLayerIdx' : 0,
    '_initialString' : 'demo_gold_single_GPU'}

def main():
    parser = argparse.ArgumentParser(description='single thread hedm reconstruction')
    parser.add_argument('-c','--config', help='config file, .yml ,.yaml, h5, hdf5', default="no config")
    parser.add_argument('-g','--gpu', help='which gpu to use(0-4)', default="0")
    args = vars(parser.parse_args())
    
    gpu=args['gpu']
    print(gpu, args['config'])
    if args['config'].endswith(('.yml','.yaml','h5','hdf5')):
        c = config.Config().load(args['config'])
        print(c)
        print(f'===== loaded external config file: {sys.argv[1]}  =====')
    else:  
        c = config.Config(**Au_Config)
        print(c)
        print('============  loaded internal config ===================')
    ################# reconstruction #########################
    try:
        S.clean_up()
    except NameError:
        pass
    S = reconstruction.Reconstructor_GPU(gpuID=gpu)  # each run should contain just one reconstructor instance, other wise GPU memory may not be released correctly.
    for i in range(1):
        S.load_config(c)
        S.serial_recon_multi_stage()
    ################# visualization #########################
    #MicFileTool.plot_mic_and_conf(S.squareMicData, 0.5)

if __name__=="__main__":
    main()
    