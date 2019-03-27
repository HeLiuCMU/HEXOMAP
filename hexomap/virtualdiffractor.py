#!/usr/bin/env python

"""
Module of components for virtual diffraction.
"""

import numpy as np

from dataclasses     import dataclass

from hexomap.orientation import Frame

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

    @staticmethod
    def load(name: str):
        """Load existing crystals from data base"""
        pass

if __name__ == "__main__":
    pass