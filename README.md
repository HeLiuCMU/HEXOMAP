# HEXOMAP: High Energy X-ray Orientation Mapping

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sry8vfFX_a9gJc084XpH12ZHFM4BPNb7#forceEdit=true&offline=true&sandboxMode=true)

[//]: # (https://colab.research.google.com/drive/1I5FUynlmLbwlF1nrRSE7bUSrRi6GVGGS#sandboxMode=true)

__HEXOMAP__ is a Cuda-based (realized through pycuda) near-filed high-energy
X-ray diffraction ([NF-HEDM](https://www.andrew.cmu.edu/user/suter/3dxdm/3dxdm.html))
reconstruction toolkit that provides 3D microstructure reconstructed with high
fadelity and efficiency.

> NOTE:  
> This GPU-based reconstruction toolkit is currently under development, and
> the API is subjected to change in the final stable realse (tentative date
> is scheduled around summer 2019).

## Installation

1. install cuda-toolkit

    *    reconmmend cuda9.1 but any version supported by pycuda is fine
2. create python virtual environment(step by step):

	* conda create --name hexomap_env_test python=3.6
	* conda activate hexomap_env_test

	* conda install numpy scipy numba matplotlib h5py jupyter
	* conda install -c anaconda yaml pyyaml dataclasses tifffile opencv
	* pip install pycuda
    
	> Or reference https://wiki.tiker.net/PyCuda/Installation/Linux/
    
    > To test installation, Type ‘python’ in terminal to enter python terminal, then try: ">>>from pycuda import autoinit"
	
2. install mpi4py:
    * conda install -c conda-forge mpi4py
	
3. Install HEXOMAP:
    * git clone https://github.com/HeLiuCMU/HEXOMAP.git
	* cd HEXOMAP/
	* python setup.py install
    
### to activate mpi usage you would need to install mpi4py. This could be tricky and varies from machine to machine. Please contact your administrator or look for documentation to install mpi4py successfully.
        
4
. check installation:
    * under HEXOMAP/: run "python -m hexomap"

## Usage and Examples
1. reconstruction	
    * see jupyter notebook: demonotebooks/*, it contains a full recontruction step ( parameter optimization and recosntruction).
    *    recon.py --config config.yml
    *    mpirun -n 4 recon_mpi.py --config config.yml
1. reduction
    *    mpirun -n 6 reduction.py
## Roadmap
* Public release with stable API by the end of summer, 2019.
* Add efficient foward NF/FF-HEDM simulation toolkit with dedicated GUI/Web frontend.

## License
__BSD 3 Cluase Licence__

## Notice
