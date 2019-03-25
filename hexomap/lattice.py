#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
General module describing crystalline material properties, including

* symmetry
* lattice

"""

import numpy as np
from dataclasses         import dataclass
from hexomap.orientation import Quaternion


@dataclass(frozen=True)
class Symmetry:
    """
    Crystal symmetry class that provides symmetry operators (in quaternion)
    and (static) method for reducing to fundamental zone

    [
        None
        cubic, bcc, fcc
        hexagonal, hcp, hex
        tetragonal, tet
        orthorhombic, ortho
    ]
    
    ref:
    https://en.wikipedia.org/wiki/Crystal_system
    https://damask.mpie.de/Documentation/CrystalLattice
    """
    lattice: str

    @property
    def sym_operator(self):
        """
        Description
        -----------
        Return a list of symmetry operator in quaternions

        Parameters
        ----------

        Returns
        -------
        list
            list of quaternions as symmetry operators
        """
        if self.lattice is None:
            return Quaternion(1,0,0,0)
        elif self.lattice.lower() in ['orthorhombic', 'ortho']:
            return [
                Quaternion(*me) for me in [
                    [ 1.0,  0.0,  0.0,  0.0 ],
                    [ 0.0,  1.0,  0.0,  0.0 ],
                    [ 0.0,  0.0,  1.0,  0.0 ],
                    [ 0.0,  0.0,  0.0,  1.0 ],
                ]
            ]
        elif self.lattice.lower() in ['tetragonal', 'tet']:
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
        elif self.lattice.lower() in ['hexagonal', 'hcp', 'hex']:
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
        elif self.lattice.lower() in ['cubic', 'bcc', 'fcc']:
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
            raise ValueError(f"Unknown lattice structure {self.lattice}")

    @staticmethod
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

    @staticmethod
    def in_fundamental_zone():
        pass

    @staticmethod
    def in_standard_stereographic_triangle():
        pass


if __name__ == "__main__":
    pass