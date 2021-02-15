#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
Configuration handler for NF-HEDM reconstruction.  By default, both YAML and 
HDF5 archive are supported.
"""

import numpy as np

from hexomap.utility import load_yaml
from hexomap.utility import write_yaml
from hexomap.utility import load_h5
from hexomap.utility import write_h5


class Config:
    """Container class for reconstruction/simulation configuration parameters"""
    
    def __init__(self, **config):
        """
        Description
        -----------
        Initialize the Config object

        Parameters
        ----------
        config: dict
            All necessary configuration parameters are represented by 
            key-value pairs

        Returns
        None
        """
        for key in config.keys():
            if isinstance(config[key],list):
                setattr(self,key,np.array(config[key]))
            else:
                setattr(self,key,config[key])

    def __repr__(self):
        """Display Configuration values."""
        return "\nConfigurations:\n" \
             + "\n".join([f"{k:30} {v}" for k,v in self.__dict__.items()])
    
    @staticmethod
    def load(fName: str='ConfigExample.yml') -> "Config":
        """
        Description
        -----------
        Generate a Config instance from yaml or hdf5 archive.

        Parameters
        ----------
        fName: str
                Name of the configure file.
        Returns
        -------
        Config
            Config instance based on given archive
        """
        print(f"Loading configuration from {fName}")
        try:
            dataMap = load_yaml(fName) if fName.endswith(('.yml','.yaml')) else load_h5(fName)
        except:
            raise TypeError("Configure file must exist and the type must be yaml or hdf5.")

        return Config(**dataMap)

    def save(self, fName: str='ConfigureFile.yml') -> None:
        """
        Description
        -----------
        Save the configuration to yaml or hdf5 file.
        Parameters
        ----------
        fName: str
                Name of the configure file.
        Returns
        -------
        None
        """
        _d = {}
        for key, val in self.__dict__.items():
            _d[key] = val.tolist() if isinstance(val, np.ndarray) else val

        try:
            write_yaml(fName, _d) if fName.endswith(('.yml', '.yaml')) else write_h5(fName, self.__dict__)
        except:
            raise IOError(f"Cannout write {fName} to disk, need to be yml or h5")

    if __name__ == "__main__":
        # testing loading yaml config file
        import os
        import hexomap
        from pprint import pprint
        exampleConfigFile = os.path.join(
            os.path.dirname(hexomap.__file__),
            "data/configs/ConfigExample.yml",
        )
        config = Config.load(exampleConfigFile)
        print(config)

        config.save('../tmp_test.yml')
        config.save('../tmp_test.h5')
