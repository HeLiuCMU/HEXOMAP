#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
General module describing crystalline material properties

NOTE:
1. Currently support lattice:
    None
    cubic, bcc, fcc
    hexagonal, hex, hcp
    orthorhombic, ortho
    tetragonal, tet
2. When comparing orientation related quantities, it is better to restrict
   them to the same referance frame, such as sample or lab frame.

"""


import numpy as np
from dataclasses         import dataclass
from hexomap.orientation import Quaternion
from hexomap.orientation import Orientation


def sym_operator(lattice:str) -> list:
    """
    Description
    -----------
    Return a list of symmetry operator in quaternions based on given lattice
    structure.

    Parameters
    ----------
    lattice: str
        lattice name

    Returns
    -------
    list
        list of quaternions as symmetry operators
    """
    if lattice is None:
        return Quaternion(1,0,0,0)
    elif lattice.lower() in ['orthorhombic', 'ortho']:
        return [
            Quaternion(*me) for me in [
                [ 1.0,  0.0,  0.0,  0.0 ],
                [ 0.0,  1.0,  0.0,  0.0 ],
                [ 0.0,  0.0,  1.0,  0.0 ],
                [ 0.0,  0.0,  0.0,  1.0 ],
            ]
        ]
    elif lattice.lower() in ['tetragonal', 'tet']:
        sqrt2 = np.sqrt(2)
        return [
            Quaternion(*me) for me in [
                [ 1.0,        0.0,        0.0,        0.0       ],
                [ 0.0,        1.0,        0.0,        0.0       ],
                [ 0.0,        0.0,        1.0,        0.0       ],
                [ 0.0,        0.0,        0.0,        1.0       ],
                [ 0.0,        0.5*sqrt2,  0.5*sqrt2,  0.0       ],
                [ 0.0,       -0.5*sqrt2,  0.5*sqrt2,  0.0       ],
                [ 0.5*sqrt2,  0.0,        0.0,        0.5*sqrt2 ],
                [-0.5*sqrt2,  0.0,        0.0,        0.5*sqrt2 ],
            ]
        ]
    elif lattice.lower() in ['hexagonal', 'hcp', 'hex']:
        sqrt3 = np.sqrt(3)
        return [
            Quaternion(*me) for me in [
                [ 1.0,        0.0,        0.0,        0.0       ],
                [-0.5*sqrt3,  0.0,        0.0,       -0.5       ],
                [ 0.5,        0.0,        0.0,        0.5*sqrt3 ],
                [ 0.0,        0.0,        0.0,        1.0       ],
                [-0.5,        0.0,        0.0,        0.5*sqrt3 ],
                [-0.5*sqrt3,  0.0,        0.0,        0.5       ],
                [ 0.0,        1.0,        0.0,        0.0       ],
                [ 0.0,       -0.5*sqrt3,  0.5,        0.0       ],
                [ 0.0,        0.5,       -0.5*sqrt3,  0.0       ],
                [ 0.0,        0.0,        1.0,        0.0       ],
                [ 0.0,       -0.5,       -0.5*sqrt3,  0.0       ],
                [ 0.0,        0.5*sqrt3,  0.5,        0.0       ],
            ]
        ]
    elif lattice.lower() in ['cubic', 'bcc', 'fcc']:
        sqrt2 = np.sqrt(2)
        return [
            Quaternion(*me) for me in [
                [ 1.0,        0.0,        0.0,        0.0       ],
                [ 0.0,        1.0,        0.0,        0.0       ],
                [ 0.0,        0.0,        1.0,        0.0       ],
                [ 0.0,        0.0,        0.0,        1.0       ],
                [ 0.0,        0.0,        0.5*sqrt2,  0.5*sqrt2 ],
                [ 0.0,        0.0,        0.5*sqrt2, -0.5*sqrt2 ],
                [ 0.0,        0.5*sqrt2,  0.0,        0.5*sqrt2 ],
                [ 0.0,        0.5*sqrt2,  0.0,       -0.5*sqrt2 ],
                [ 0.0,        0.5*sqrt2, -0.5*sqrt2,  0.0       ],
                [ 0.0,       -0.5*sqrt2, -0.5*sqrt2,  0.0       ],
                [ 0.5,        0.5,        0.5,        0.5       ],
                [-0.5,        0.5,        0.5,        0.5       ],
                [-0.5,        0.5,        0.5,       -0.5       ],
                [-0.5,        0.5,       -0.5,        0.5       ],
                [-0.5,       -0.5,        0.5,        0.5       ],
                [-0.5,       -0.5,        0.5,       -0.5       ],
                [-0.5,       -0.5,       -0.5,        0.5       ],
                [-0.5,        0.5,       -0.5,       -0.5       ],
                [-0.5*sqrt2,  0.0,        0.0,        0.5*sqrt2 ],
                [ 0.5*sqrt2,  0.0,        0.0,        0.5*sqrt2 ],
                [-0.5*sqrt2,  0.0,        0.5*sqrt2,  0.0       ],
                [-0.5*sqrt2,  0.0,       -0.5*sqrt2,  0.0       ],
                [-0.5*sqrt2,  0.5*sqrt2,  0.0,        0.0       ],
                [-0.5*sqrt2, -0.5*sqrt2,  0.0,        0.0       ],
            ]
        ]
    else:
        raise ValueError(f"Unknown lattice structure {lattice}")


def vec_in_standard_stereographic_triangle(vec: np.ndarray, 
                                           lattice: str,
                                        ) -> bool:
    """
    Description
    -----------
    Check whether a given vector will project to the standard stereographic
    triangle, useful for computing Inverse Pole Figure color tuple.
        ---------
        !Warning!
        ---------
        The calculation performed here is assuming that the vector is in
        the lattice framework.
    
    Parameters
    ----------
    vec: np.ndarray
        target vector
    lattice: str
        lattice name
    
    Returns
    -------
    bool
    """
    # Considers only vectors with z >= 0, hence uses two neighboring SSTs.
    vec = vec * np.sign(vec[2])


def in_fundamental_zone(o: "Orientation", lattice: str) -> bool:
    """
    Description
    -----------
    Chekc if the orientation is in its fundamental zone by checking its
    Rodrigues representation.

    Parameter
    ---------
    o: Orientation
        Orientation represents a certain attitude
    lattice: str
        Lattice symmetry

    Returns
    -------
    bool

    NOTE:
        migrated from DAMASK.orientation module
    """
    r = np.absolute(o.as_rodrigues.as_array)
    # NOTE:
    #  The following comparison is to evaluate 
    if lattice.lower() in ['cubic', 'bcc', 'fcc']:
        sqrt2 = np.sqrt(2)
        return  sqrt2-1.0 >= r[0] \
            and sqrt2-1.0 >= r[1] \
            and sqrt2-1.0 >= r[2] \
            and 1.0 >= r[0] + r[1] + r[2]
    elif lattice.lower() in ['hexagonal', 'hex', 'hcp']:
        sqrt3 = np.sqrt(3)
        return  1.0 >= r[0] and 1.0 >= r[1] and 1.0 >= r[2] \
            and 2.0 >= sqrt3*r[0] + r[1] \
            and 2.0 >= sqrt3*r[1] + r[0] \
            and 2.0 >= sqrt3 + r[2]
    elif lattice.lower() in ['tetragonal', 'tet']:
        sqrt2 = np.sqrt(2)
        return  1.0 >= r[0] and 1.0 >= r[1] \
            and sqrt2 >= r[0] + r[1] \
            and sqrt2 >= r[2] + 1.0
    elif lattice.lower() in ['orthorhombic', 'orth']:
        return  1.0 >= r[0] and 1.0 >= r[1] and 1.0 >= r[2]
    else:
        return True


def in_standard_stereographic_triangle(o: 'Orientation',
                                       lattice: str,
                                    ) -> bool:
    """
    Description
    -----------
    Check if given orientation lies in the standard stereographic
    triangle of given lattice.

    ref:
        Representation of Orientation and Disorientation Data for Cubic, 
        Hexagonal, Tetragonal and Orthorhombic Crystals
        Acta Cryst. (1991). A47, 780-789
    
    Parameters
    ----------
    o: Orientation
        input orientation instance
    lattice: str
        lattice name

    Returns
    -------
    bool

    NOTE
    ----
    This function is adapted from the orientation module in DAMASK
    (damask.mpie.de) 
    """
    r = o.as_rodrigues.as_array

    if lattice.lower() in ['cubic', 'bcc', 'fcc']:
        return  r[0] >= r[1] \
            and r[1] >= r[2] \
            and r[2] >= 0.0
    elif lattice.lower() in ['hexagonal', 'hcp', 'hex']:
        sqrt3 = np.sqrt(3)
        return  r[0] >= sqrt3*r[1]\
            and r[1] >= 0 \
            and r[2] >= 0
    elif lattice.lower() in ['tetragonal', 'tet']:
        return  r[0] >= r[1] \
            and r[1] >= 0 \
            and r[2] >= 0
    elif lattice.lower() in ['orthorhombic', 'ortho']:
        return  r[0] >= 0 \
            and r[1] >= 0 \
            and r[2] >= 0
    else:
        return True


if __name__ == "__main__":
    pass