import numpy as np
import h5py
import yaml
class Config():
    def __init__(self, **config):
        """
        Description
        -----------
        Initialize the Config object

        Parameters
        ----------
        config: dict
                All necessary configuration parameters are represented by hey-value pairs

        Returns
        -------
        None
        """
        for key in config.keys():
            if isinstance(config[key],list):
                setattr(self,key,np.array(config[key]))
            else:
                setattr(self,key,config[key])

    def __repr__(self):
        """Display Configuration values."""
        res="\nConfigurations:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                res=res+"{:30} {}\n".format(a, getattr(self, a))
        return res
    
    @classmethod
    def load(cls,fName='ConfigExample.yml',fType=None):
        """
        Description
        -----------
        Generate a Config object from yaml or hdf5 file.

        Parameters
        ----------
        fName: str
                Name of the configure file.
        fType: str
                'yml' ('yaml') or 'h5' ('hdf5')

        Returns
        -------
        Config object
        """
        print('\n Loading configuration from %s'%fName)
        if fName.endswith(('.yml','.yaml')) or fType in ['yml','yaml']:
            try:
                with open(fName,'r') as f:
                    dataMap=yaml.safe_load(f)
                    return Config(**dataMap)
            except IOError as e:
                print("Couldn't open configuration file\n (%s)."%e)
        elif fName.endswith(('.h5','.hdf5')) or fType in ['h5','hdf5']:
            try:
                with h5py.File(fName,'r') as f:
                    dataMap=cls._recursively_load_dict_contents_from_group(f,'/')
                    return Config(**dataMap)
            except IOError as e:
                print("Couldn't open configuration file\n (%s)."%e)
        else:
            raise IOError("Configure file type must be yaml or hdf5.")


    def save(self,fName='ConfigureFile.yml',fType=None):
        """
        Description
        -----------
        Save the configuration to yaml or hdf5 file.

        Parameters
        ----------
        fName: str
                Name of the configure file.
        fType: str
                'yml' ('yaml') or 'h5' ('hdf5')

        Returns
        -------
        None
        """
        if fName.endswith(('.h5','.hdf5')) or fType in ('h5','hdf5'):
            if not fName.endswith(('.h5','.hdf5')):
                fName=fName+".h5"
            print('\n Saving configuration to %s'%fName)

            d = {}
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)):
                    d[a] = getattr(self, a)

            try:
                with h5py.File(fName,'w') as f:
                    self._recursively_save_dict_contents_to_group(f,'/',d)
            except IOError as e:
                print("%s\n"%e)

        elif fName.endswith(('.yml','.yaml')) or fType in ['yml','yaml']:
            if not fName.endswith(('.yml','.yaml')):
                fName=fName+".yml"
            print('\n Saving configuration to %s'%fName)

            d = {}
            for a in dir(self):
                if not a.startswith("__") and not callable(getattr(self, a)):
                    item=getattr(self,a)
                    if isinstance(item,np.ndarray):
                        d[a]=item.tolist()
                    else:
                        d[a] =item

            try:
                with open(fName,'w') as f:
                    yaml.dump(d, f, default_flow_style=False)
            except IOError as e:
                print("%s\n"%e)

        else:
            raise IOError("Configure file type need to be yaml or hdf5.")
        
        
    @classmethod
    def _recursively_load_dict_contents_from_group(cls, h5file, path):
        """
        ....
        """
        ans = {}
        for key, item in h5file[path].items():
            if isinstance(item, h5py._hl.dataset.Dataset):
                ans[key] = item.value
            elif isinstance(item, h5py._hl.group.Group):
                ans[key] = cls.recursively_load_dict_contents_from_group(h5file, path + key + '/')
        return ans


    def _recursively_save_dict_contents_to_group(self, h5file, path, dic):
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

