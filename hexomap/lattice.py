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
from dataclasses           import dataclass
from hexomap.npmath        import normalize
from hexomap.orientation   import Quaternion
from hexomap.orientation   import Orientation
from hexomap.orientation   import sym_operator


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


def to_fundamental_zone(o: 'Orientation', lattice: str) -> "Orientation":
    """
    Description
    -----------
    Reduce the orientation to the fundamental zone with given lattice symmetry

    Parameters
    ----------
    o: Orientation
        Input orientation

    lattice: str
        Lattice name
    
    Returns
    -------
    Orientation
        Orientation with attitude expressed in the fundamental zone of given
        lattice symmetry type
    """
    _q = Quaternion(*o.q.as_array)
    for symop in sym_operator(lattice):
        o.q = _q*symop
        if in_fundamental_zone(o, lattice):
            return o


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


def calc_inverse_pole_figure_color(o: 'Orientation',
                                   p: np.ndarray,
                                   lattice: str,
                                ) -> tuple:
    """
    Description
    -----------
        Calculate the Inverse Pole Figure (IPF) color tuple common used to
        represent the crystal orientations in an EBSD map.
        The orientation and the pole must be associated with the same reference
        frame.

        !WARNING!
            This function does not provide frame check.  It is the users
            responsibility to ensure that the orientation and the pole are
            associated with the same frame.

    Parameter
    ---------
    o: Orientation
        Crystal orientation, preferably associated with the sample frame
    p: np.ndarray
        Pole direction, often expressed in the sample frame, such as
            ND: sample normal directon
            RD: sample rolling direction
            TD: transverse direction (or tensile direction)
    lattice: str
        Lattice name
    
    Returns
    -------
    tuple
        RGB color tuple with range between 0 and 1 
    """
    # convert the pole from the common (sample) frame to the crystal frame
    rot_m = o.as_matrix     # active rotation matrix
    p = np.dot(rot_m.T, p)  # bring pole to the crystal frame
    p = normalize(p)

    # generate the three reference pole (crystal frame) in the IPF triangle
    basis = get_inverse_pole_figure_ref_poles(lattice)

    # use the coefficient as the RGB IPF color tuple
    #              basis                ipf          p
    #              -----                ---          -
    #     | red_1  green_1  blue_1 | | ipf_r |  = | p_1 |
    #     | red_2  green_2  blue_2 | | ipf_g |  = | p_2 |
    #     | red_3  green_3  blue_3 | | ipf_b |  = | p_3 |
    #
    for sym_opt in sym_operator(lattice):
        p = Quaternion.quatrotate(sym_opt, p)
        ipf = np.dot(np.linalg.inv(basis), p)
        if np.all(ipf < 0):
            continue  # not in the SST, check next equivalent orientation
        else:
            ipf = np.minimum(normalize(ipf)**0.5, np.ones(3))
            return ipf/ipf.max()
    # if we cannot find any, something is seriously wrong
    raise ValueError("Cannot find equivalent orientation in SST")


def get_inverse_pole_figure_ref_poles(lattice: str):
    """
    Description
    -----------
        Return the reference poles (crystal frame) for given lattice
    
    Parameters
    ----------
    lattice: str
        lattice name
    
    Returns
    -------
    np.ndarray
        basis in crystal frame
    """
    if lattice.lower() in ['cubic', 'bcc', 'fcc']:
        basis = np.array([
            [0, 0, 1],   # red pole
            [1, 0, 1],   # green pole
            [1, 1, 1],   # blue pole
        ])
    elif lattice.lower() in ['hexagonal', 'hcp', 'hex']:
        basis = np.array([
            [0,              0, 1],  # c-axis, [0001], red
            [1,              0, 0],  # a-axis, [2-1-10], green
            [np.sqrt(3)/2, 0.5, 0],  # [10-10], blue
        ])
    elif lattice.lower() in ['tetragonal', 'tet']:
        basis = np.array([
            [0, 0, 1],  # red
            [1, 0, 0],  # green
            [1, 1 ,0],  # blue
        ])
    elif lattice.lower() in ['orthorhombic', 'ortho']:
        basis = np.array([
            [0, 0, 1],  # red
            [1, 0, 0],  # green
            [0, 1, 0],  # blue
        ])
    else:
        raise ValueError(f"Unknown lattice {lattice}")
    return normalize(basis.T, axis=0)


def basis_lattice(lattice: str) -> np.ndarray:
    """
    Description
    -----------
        Return a matrix whose column space is the three lattice vector
            m = | a_1  b_1  c_1 |
                | a_2  b_2  c_2 |
                | a_3  b_3  c_3 |
        the covariances (a|b|c_i) are in the crystal frame

    Parameters
    ----------
    lattice: str
        lattice name

    Returns
    -------
    np.ndarray
        column space matrix
    """
    pass


def reciprocal_basis_lattice(lattice: str) -> np.ndarray:
    """
    """
    pass


if __name__ == "__main__":
    print("testing")
