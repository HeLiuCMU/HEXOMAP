# HEXOMAP: High Energy X-ray Orientation Mapping

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sry8vfFX_a9gJc084XpH12ZHFM4BPNb7#forceEdit=true&offline=true&sandboxMode=true)


<!--(https://colab.research.google.com/drive/1I5FUynlmLbwlF1nrRSE7bUSrRi6GVGGS#sandboxMode=true)--!>

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
1. create python virtual environment

    *   insall from environment file:

        *	install anaconda

        * conda env create -f environment.yml
        * under HEXOMAP/: python setup.py install
        
1. check installation:
    * under HEXOMAP/: run "python -m hexomap"

## Usage and Examples
1. reconstruction
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
