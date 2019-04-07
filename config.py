import numpy as np
import h5py
import yaml
class Config(object):
    def __init__(self,configure_file_name='ConfigExample.yml'):
        """Set values of computed attributes."""
        self.__name =  'name'
        try:
            with open(configure_file_name,'r') as f:
                dataMap=yaml.safe_load(f)
        except IOError as e:
            print("Couldn't open file (%s)."%e)
        micsize = np.array(dataMap['ReconParam']['micsize'])
        micVoxelSize = dataMap['ReconParam']['micVoxelSize']
        micShift = np.array(dataMap['ReconParam']['micShift'])
        expdataNDigit = dataMap['InOut_Files']['expdataNDigit']
        energy = dataMap['Setup']['energy'] # in kev
        sample = dataMap['Material_Name']
        maxQ = dataMap['ReconParam']['maxQ']
        etalimit = dataMap['ReconParam']['etalimit']
        NRot = dataMap['Setup']['NRot']
        NDet = dataMap['Setup']['NDet']
        searchBatchSize = dataMap['ReconParam']['searchBatchSize']
        reverseRot=dataMap['Setup']['reverseRot']          # for aero, is True, for rams: False
        detL = np.array(dataMap['Setup']['detL'])
        detJ = np.array(dataMap['Setup']['detJ']) 
        detK = np.array(dataMap['Setup']['detK'])
        detRot = np.array(dataMap['Setup']['detRot'])
        detNJ = np.array(dataMap['Setup']['detNJ'])
        detNK=np.array(dataMap['Setup']['detNK'])
        detPixelJ=np.array(dataMap['Setup']['detPixelJ'])
        detPixelK=np.array(dataMap['Setup']['detPixelK'])
        fileBin = dataMap['InOut_Files']['fileBin']
        fileBinDigit = dataMap['InOut_Files']['fileBinDigit']
        fileBinDetIdx = np.array(dataMap['InOut_Files']['fileBinDetIdx'])
        fileBinLayerIdx = dataMap['InOut_Files']['fileBinLayerIdx']
        fileFZ = dataMap['fileFZ']
        _initialString = dataMap['InOut_Files']['initialString']


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
