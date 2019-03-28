#!/usr/bin/env python

"""
Module of components for virtual diffraction.
"""

import os
import yaml
import numpy as np
from dataclasses         import dataclass
from itertools           import product
from hexomap.orientation import Frame
from hexomap.npmath      import norm
from hexomap.utility     import iszero

# -- Define standard frames commmonly used for NF/FF-HEDM --
STD_FRAMES = {
    'APS': Frame(
        e1=np.array([ 1, 0, 0]),  # APS_X
        e2=np.array([ 0, 1, 0]),  # APS_Y
        e3=np.array([ 0, 0, 1]),  # APS_Z
        o =np.array([ 0, 0, 0]),  # rotation stage center
        name='aps'
        ),
    "Detector": Frame(
        e1=np.array([-1, 0, 0]),  # detector_j
        e2=np.array([ 0,-1, 0]),  # detector_k
        e3=np.array([ 0, 0, 1]),  # detector_n, norm
        o =np.array([ 0, 0, 5]),  # rotation stage center, assuming 5mm detector distance
        name='detector_1'
    ),
}

# -- Define the materials data folder direction
DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
MATS_DIR = os.path.join(DATA_DIR, 'materials')

@dataclass
class Detector:
    frame:     "Frame" = STD_FRAMES["Detector"]
    resolution: tuple  = (2048, 2048)           # number of pixels
    pix_size:   tuple  = (0.00148, 0.00148)     # mm or m?

    # Move
    def transform_detector(self, m: np.ndarray) -> None:
        """
        Description
        -----------
        Transfer detector frame using given transformation matrix.

        Parameters
        ----------
        m: np.ndarray, (4, 4)
            Transformation matrix containing both translation and rotation
        
        Returns
        -------
        None
        """
        pass

    # IntersectoinIdx
    def acquire_signal(self, 
                       scatter_vec: np.ndarray, 
                       bragg_angle: float,
                       eta: float,
                    ) -> tuple:
        """
        Description
        -----------

        Parameters
        ----------

        Returns
        -------
        """
        pass

    # BackProj
    def back_projection(self,
                        signal_position: tuple,  # (J,K) in pixels
                        omega: float,
                        bragg_angle: float,
                        eta: float,
                        target_frame: "Frame"
                    ) -> tuple:
        """
        """
        pass


@dataclass
class Crystal:
    name:  str
    atoms: list
    atomz: list
    lattice: str
    lattice_constant: list

    def __post_init__(self):
        # construct the unit cell (prism) for given crystal
        self.prism = Crystal.prism_from_lattice_constant(self.lattice_constant)

    def structure_factor(self, hkl):
        """Calculate structure factor"""
        return np.dot(self.atomz,
                      np.exp(-2*np.pi*1j*np.dot(np.array(self.atoms), np.array(hkl).reshape((3, 1)))),
        )
    
    def scatter_vecs(self, q_max: int) -> list:
        """Generate scattering vectors with Eward sphere capped at q_max"""
        recip_prism = Crystal.prism_to_reciprocal(self.prism)
        h_max, k_max, l_max = (q_max/norm(recip_prism, axis=0)).astype(int)
        hkls = product(range(-h_max, h_max+1), 
                       range(-k_max, k_max+1), 
                       range(-l_max, l_max+1),
                    )
        return [
            np.dot(recip_prism, hkl) 
                for hkl in hkls
                if not iszero(sum(map(abs, hkl)))                 # hkl != [000]
                if norm(hkl) <= q_max                             # within Eward sphere
                if not iszero(self.structure_factor(hkl))         # non-vanishing
        ]
    
    @staticmethod
    def load(element:str, name: str) -> 'Crystal':
        """
        Description
        -----------
            Load material config for given materials from data base
        
        Parameters
        ----------
        element: str
            main element, for example, titanium for Ti64
        name: str
            abbreviation for target material, for example Ti64 for Ti-6Al-4V

        Returns
        -------
        Crystal
        """
        with open(os.path.join(MATS_DIR, f"{element}.yml"), 'r') as f:
            mat_dict = yaml.safe_load(f)['variants'][name]
        
        return Crystal(
            name,
            [me['pos'] for me in mat_dict['atoms']],
            [me['atomic_number'] for me in mat_dict['atoms']],
            mat_dict['crystal_structure'],
            [val for _, val in mat_dict['lattice_constant'].items()]
            )
    
    @staticmethod
    def prism_from_lattice_constant(lattice_constant: list, 
                                    in_degrees=True,
                                ) -> np.ndarray:
        """
        Description
        -----------
            Calculate the unit cell prism expressed in crystal Frame

        Parameters
        ----------
        lattice_constat: list
            lattice constants for target crystal
        in_degrees: bool
            unit of alpha, beta, gamma in lattice constants

        Returns
        -------
        np.ndarray
            column-stacked base vectors for the unit cell prism expressed in
            crystal frame
        """
        a, b, c, alpha, beta, gamma = lattice_constant
        if in_degrees:
            alpha, beta, gamma = np.radians([alpha, beta, gamma])
        # compute unit cell from lattice constants
        # ref:
        # https://github.com/KedoKudo/cyxtal/blob/master/documentation/dev/development.pdf
        c_a, c_b, c_g = np.cos([alpha, beta, gamma])
        s_g = np.sin(gamma)
        factor = 1 + 2*c_a*c_b*c_g - c_a**2 - c_b**2 - c_g**2
        v_cell = a*b*c*np.sqrt(factor)
        v1 = [a, 0, 0]
        v2 = [b*c_g, b*s_g, 0.0]
        v3 = [c*c_b, c*(c_a-c_b*c_g)/(s_g), v_cell/(a*b*s_g)]
        return np.column_stack((v1,v2,v3))
    
    @staticmethod
    def prism_to_reciprocal(prism: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
            Calcualte the reciprocal dual of given prism (column stacked)
            
            ref:
            https://en.wikipedia.org/wiki/Reciprocal_lattice
        
        Parameters
        ----------
        prism: np.ndarray
            unit cell prism

        Returns
        -------
        np.ndarray
            Reciprocal dual of the unit cell prism

        NOTE:
        use pinv to avoid singular matrix from ill-positioned problem
        """
        return np.transpose(2*np.pi*np.linalg.pinv(prism))


# TODO:
# Finish the collector after the Detector and Crystal class refactor is complete
def collect_virtual_patterns(detector: 'Detector',
                             xtal: 'Crystal',
                        ):
    """
    Generate list of peaks (HEDM patterns) for given crystal(sample) on the target detector
    """
    pass


if __name__ == "__main__":
    # example_1: 
    xtal = Crystal.load('gold', 'gold_fcc')
    print(xtal.prism)
    print(Crystal.prism_to_reciprocal(xtal.prism))
    print(norm(Crystal.prism_to_reciprocal(xtal.prism), axis=0))
    print(xtal.scatter_vecs(3))
