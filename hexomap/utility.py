#!/usr/bin/env python

"""
Utility module for hexomap package.
"""

import yaml
import h5py
import numpy as np

from functools import singledispatch
from functools import update_wrapper


def load_kernel_code(filename):
    """return the cuda source code"""
    with open(filename, 'r') as f:
        kernel_code = f.read()
    return kernel_code


def methdispatch(func):
    """class method overload decorator"""
    # ref:
    #   https://stackoverflow.com/questions/24601722/how-can-i-use-functools-singledispatch-with-instance-methods
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def isone(a: float) -> bool:
    """Work around with float precision issues"""
    return np.isclose(a, 1.0, atol=1.0e-8, rtol=0.0)


def iszero(a: float) -> bool:
    """Work around with float precision issues"""
    return np.isclose(a, 0.0, atol=1.0e-12, rtol=0.0)


def standarize_euler(euler: np.ndarray, in_radian=True) -> np.ndarray:
    """
    Force Euler angle to be betewen 
        [0~2pi, 0~pi, 0~2pi]
    """
    if not in_radian:
        euler = np.radians(euler)
    return np.where(
        euler<0, 
        (euler+2.0*np.pi)%np.array([2.0*np.pi,np.pi,2.0*np.pi]),
        euler%(2*np.pi)
        )


def load_yaml(fname: str) -> dict:
    """generic safe_load wrapper for yaml archive"""
    try:
        with open(fname, 'r') as f:
            dataMap = yaml.safe_load(f)
    except IOError as e:
        print(f"Cannot open YAML file {fname}")
        print(f"IOError: {e}")
    
    return dataMap


def write_yaml(fname: str, data: dict) -> None:
    """generic output handler for yaml archive"""
    try:
        with open(fname, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    except IOError as e:
        print(f"Cannot write YAML file {fname}")
        print(f"IOError: {e}")


def write_h5(fname: str, data: dict) -> None:
    """generic simper HDF5 archive writer"""
    try:
        with h5py.File(fname, 'w') as f:
            recursively_save_dict_contents_to_group(f,'/',data)
    except IOError as e:
        print(f"Cannot write HDF5 file {fname}")
        print(f"IOError: {e}")


def load_h5(fname: str, path: str='/') -> dict:
    """generic simple HDF5 archive loader"""
    try:
        with h5py.File(fname, 'r') as f:
            dataMap = recursively_load_dict_contents_from_group(f, path)
    except IOError as e:
        print(f"Cannot open HDF5 file {fname}")
        print(f"IOError: {e}")

    return dataMap

def print_h5(fname: str) -> None:
    """generic simple HDF5 archive structure printer"""
    try:
        with h5py.File(fname, 'r') as h:
            print(fname)
            recursively_print_structure(h, '  ')
    except IOError as e:
        print(f"Cannot open HDF5 file {fname}")
        print(f"IOError: {e}")

def recursively_print_structure(item, leading = ''):
    """recusively print HDF5 archive structure"""
    for key in item:
        if isinstance(item[key], h5py.Dataset):
            print(leading + key + ': ' + str(item[key].shape))
        else:
            print(leading + key)
            recursively_print_structure(item[key], leading + '  ')


def recursively_load_dict_contents_from_group(h5file: "h5py.File", 
                                              path: str,
        ) -> dict:
    """recursively load data from HDF5 archive as dict"""
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, f"{path}{key}/")
    return ans


def recursively_save_dict_contents_to_group(h5file: "h5py.File", 
                                            path: str, 
                                            dic: dict,
        ) -> None:
    """recursively write data to HDF5 archive"""
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes,int,float,np.bool_)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError(f'Cannot save {item} type')
            
if __name__ == "__main__":
    pass
