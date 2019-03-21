import numpy as np
import h5py
class Config(object):
    micsize = np.array([150, 150])
    micVoxelSize = 0.007
    micShift = np.array([0.0, 0.0, 0.0])
    expdataNDigit = 6
    energy = 55.587 # in kev
    sample = 'gold'
    maxQ = 8
    etalimit = 81 / 180.0 * np.pi
    NRot = 180
    NDet = 2
    searchBatchSize = 10000
    reverseRot=True          # for aero, is True, for rams: False
    detL = np.array([[ 8.9587, 10.9587]])
    detJ = np.array([[1020.4672, 1035.6063]]) 
    detK = np.array([[1995.8868, 1990.6409]])
    detRot = np.array([[[89.0107, 91.0776, -0.4593],
                  [88.8744, 90.8149, -0.4466]]])
    detNJ = np.array([2048,2048, 2048, 2048])
    detNK=np.array([2048, 2048, 2048, 2048])
    detPixelJ=np.array([0.00148,0.00148,0.00148,0.00148])
    detPixelK=np.array([0.00148,0.00148,0.00148,0.00148])
    fileBin = 'binary_file_initial'
    fileBinDigit = 6
    fileBinDetIdx = np.array([0, 1])
    fileBinLayerIdx = 0
    fileFZ = 'fz file'
    _initialString = 'test_initial'

    def __init__(self):
        """Set values of computed attributes."""
        self.__name =  'name'

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
    def save(self, fName, fType='h5'):
        '''
        save this config
        '''
        print("\nsaving configuration")
        d = {}
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                d[a] = getattr(self, a)
        #print(d)
        self.save_dict_to_hdf5(d, fName)
        print("\n")
    def load(self, fName):
        '''
        load config
        
        '''
        dic = self.load_dict_from_hdf5(fName)
        for k, v in dic.items():
            setattr(self, k, v)
        
    def save_dict_to_hdf5(self, dic, filename):
        """
        ....
        """
        with h5py.File(filename, 'w') as h5file:
            self.recursively_save_dict_contents_to_group(h5file, '/', dic)

    def recursively_save_dict_contents_to_group(self, h5file, path, dic):
        """
        ....
        """
        for key, item in dic.items():
            if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes,int,float,np.bool_)):
                h5file[path + key] = item
            elif isinstance(item, dict):
                self.recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
            else:
                raise ValueError('Cannot save %s type'%type(item))

    def load_dict_from_hdf5(self, filename):
        """
        ....
        """
        with h5py.File(filename, 'r') as h5file:
            return self.recursively_load_dict_contents_from_group(h5file, '/')

    def recursively_load_dict_contents_from_group(self, h5file, path):
        """
        ....
        """
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = self.recursively_load_dict_contents_from_group(h5file, path + key + '/')
        return ans