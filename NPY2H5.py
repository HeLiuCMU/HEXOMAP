import numpy as np
import h5py

def h5printR(item, leading = ''):
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            h5printR(item[key], leading + '  ')

# Print structure of a `.h5` file            
def h5print(filename):
    with h5py.File(filename, 'r') as h:
        print(filename)
        h5printR(h, '  ')
        
def npy2h5(fName, h5Name, material,lLayerIdx, q=11):
    '''
    lFName: list of filenames
    lLayerIdx: list of layer index
    '''
    with h5py.File(h5Name,'w') as fout:
        md=fout.create_group('meta_data')
        sls=fout.create_group('slices')
        md.create_dataset('material',data=np.string_(material))
        md.create_dataset('maxQ',data=np.int_(q))
        for i,f in enumerate(lFName):
            print(i,f)
            grp=sls.create_group('z{:d}'.format(lLayerIdx[i]))
            a=np.load(f)
            ds=grp.create_dataset('x',data=a[:,:,0]*1000,dtype='float32')
            ds.attrs['info']='X coordinate (micron meter).'
            ds=grp.create_dataset('y',data=a[:,:,1]*1000,dtype='float32')
            ds.attrs['info']='Y coordinate (micron meter).'
            ds=grp.create_dataset('EulerAngles',data=a[:,:,3:6]*np.pi/180,dtype='float32')
            ds.attrs['info']='active ZXZ Euler angles (radian)'
            ds=grp.create_dataset('phase',data=a[:,:,7],dtype='uint16')
            ds.attrs['info']='material phases'
            ds=grp.create_dataset('Confidence',data=a[:,:,6],dtype='float32')
            ds.attrs['info']='hit ratio of simulated peaks and experimental peaks'
    print('=== saved format:')
    h5print(h5Name)
if __name__ == "__main__":
    lFName = [f'dummy_2_single_crystal_furnace_nf_copperBCC_q11_rot720_z{layer}_150x150_0.007_shift_0.0_0.0_0.0.npy' for layer in range(3)]
    print(lFName)
    h5Name = f'dummy_2_single_crystal_furnace_nf_copperBCC_q11_rot720_3layers_150x150_0.007_shift_0.0_0.0_0.0.h5'
    print(h5Name)
    lLayerIdx = np.arange(3)
    npy2h5(lFName, h5Name, 'copperBCC', lLayerIdx,q=11)