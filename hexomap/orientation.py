#!/usr/bin/env python
# -*- coding: UTF-8 no BOM -*-

"""
General math module for crystal orientation related calculation.

Most of the conventions used in this module is based on:
    D Rowenhorst et al. 
    Consistent representations of and conversions between 3D rotations
    10.1088/0965-0393/23/8/083501

with the exceptions:
    1. An orientation is always attached to a frame, and all calculations
       between orientations can only be done when all of them are converted
       to the same frame.
    2. Always prefer SI units.

Conversion chain:
    Rodrigues (angle,axis) <-> quaternion <-> Euler angles(ZXZ) <-> rotation matrix
"""

import numpy              as np
import concurrent.futures as cf

from dataclasses     import dataclass
from typing          import Union
from hexomap.npmath  import norm
from hexomap.npmath  import normalize
from hexomap.npmath  import random_three_vector
from hexomap.utility import methdispatch
from hexomap.utility import iszero
from hexomap.utility import isone
from hexomap.utility import standarize_euler

@dataclass
class Eulers:
    """
    Euler angles representation of orientation.

    phi1: [0, 2pi]
    phi:  [0, pi]
    phi2: [0, 2pi]

    Euler angle definitions:
        'Bunge' :  z -> x -> z     // prefered
    """
    phi1: float  # [0, 2pi)
    phi:  float  # [0,  pi]
    phi2: float  # [0, 2pi)
    in_radians: bool=True
    order: str='zxz'
    convention: str='Bunge'

    def __post_init__(self):
        # force euler to the standard range
        _euler = np.array([self.phi1, self.phi, self.phi2])
        self.phi1, self.phi, self.phi2 = standarize_euler(_euler, 
                                                          self.in_radians,
                                                        )
        self.in_radians = True

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.phi1, self.phi, self.phi2])

    @property
    def as_matrix(self):
        """
        Return the active rotation matrix
        """
        # NOTE:
        #   It is not recommended to directly associated Euler angles with
        #   other common transformation concept due to its unique passive
        #   nature.
        #   However, I am providing the conversion to (orientation) matrix
        #   here for some backward compatbility.
        c1, s1 = np.cos(self.phi1), np.sin(self.phi1)
        c,  s  = np.cos(self.phi ), np.sin(self.phi )
        c2, s2 = np.cos(self.phi2), np.sin(self.phi2)
        return np.array([
            [ c1*c2-s1*c*s2, -c1*s2-s1*c*c2,  s1*s],
            [ s1*c2+c1*c*s2, -s1*s2+c1*c*c2, -c1*s],
            [          s*s2,           s*c2,     c],
        ])
    
    @staticmethod
    def from_matrix(m: np.ndarray):
        """
        Description
        -----------
            Initialize an Euler angle with a given rotation matrix
        
        Parameters
        ----------
        m: np.ndarray
            input rotation matrix
        
        Returns
        -------
        Eulers
        """
        if isone(m[2,2]**2):
            return Eulers(
                0.0,
                0.0,
                np.arctan2(m[1,0], m[0,0]), 
            )
        else:
            return Eulers(
                np.arctan2(m[0,2], -m[1,2]),
                np.arccos(m[2,2]),
                np.arctan2(m[2,0], m[2,1]),
            )

    @staticmethod
    def eulers_to_matrices(eulers: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
            Vectorized batch conversion from Eulers (Bunge) angles to rotation
            matices
        
        Parameters
        ----------
        eulers: np.ndarray
            euler angles with the shape of (n_eulers, 3)
        
        Returns
        -------
        np.ndarray
            rotation matrices representation of the input Euler angles with
            the shape of (n_eulers, 3, 3)

        NOTE
        ----
        Testing with 10k eulers
            original implementation
        >>%timeit m_old = EulerZXZ2MatVectorized(eulers)
        1.17 ms ± 10.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        >>%timeit m_new = Eulers.eulers_to_matrices(eulers)
        1.2 ms ± 22.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        """
        # ensure shape is correct
        try:
            eulers = eulers.reshape((-1, 3))
        except:
            raise ValueError(f"Eulers angles much be ROW/horizontal stacked")
        
        c1, s1 = np.cos(eulers[:,0]), np.sin(eulers[:,0])
        c,  s  = np.cos(eulers[:,1]), np.sin(eulers[:,1])
        c2, s2 = np.cos(eulers[:,2]), np.sin(eulers[:,2])
        
        m = np.zeros((eulers.shape[0], 3, 3))
        m[:,0,0], m[:,0,1], m[:,0,2] = c1*c2-s1*c*s2, -c1*s2-s1*c*c2,  s1*s
        m[:,1,0], m[:,1,1], m[:,1,2] = s1*c2+c1*c*s2, -s1*s2+c1*c*c2, -c1*s
        m[:,2,0], m[:,2,1], m[:,2,2] =          s*s2,           s*c2,     c

        return m
    
    @staticmethod
    def matrices_to_eulers(matrices: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
            Vectorized batch conversion from stack of rotation matrices to 
            Euler angles (Bunge)

        Parameter
        ---------
        matrices: np.ndarray
            stack of rotation matrices

        Returns
        -------
        np.ndarray
            stakc of Euler angles (Bunge)

        Note
        ----
        Testing with 10k rotation matrices
            original implementation
        >>%timeit eulers_o = Mat2EulerZXZVectorized(Rs) 
        2.01 ms ± 87.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
            current implementation
        >>%timeit eulers_n = Eulers.matrices_to_eulers(Rs)
        1.45 ms ± 8.63 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
        """
        try:
            matrices = matrices.reshape((-1,3,3))
        except:
            raise ValueError("Please stack rotation matrices along 1st-axis")
        
        eulers = np.zeros((matrices.shape[0], 3))

        # first work the degenerated cases
        _idx = np.isclose(matrices[:,2,2]**2, 1)
        eulers[_idx, 2] =  np.arctan2(matrices[_idx,1,0], matrices[_idx,0,0]) 
        # then the general cases
        _idx = (1 - _idx).astype(bool)
        eulers[_idx, 0] = np.arctan2(matrices[_idx,0,2], -matrices[_idx,1,2])
        eulers[_idx, 1] = np.arccos(matrices[_idx,2,2]),
        eulers[_idx, 2] = np.arctan2(matrices[_idx,2,0], matrices[_idx,2,1]),

        return eulers%(2*np.pi)


@dataclass
class Rodrigues:
    """
    Rodrigues–Frank vector: ([n_1, n_2, n_3], tan(ω/2))
    """
    r1: float
    r2: float
    r3: float

    @property
    def as_array(self) -> np.ndarray:
        """As numpy array"""
        return np.array([self.r1, self.r2, self.r3])

    @property
    def rot_axis(self) -> np.ndarray:
        """Rotation axis"""
        return normalize(self.as_array)

    @property
    def rot_ang(self) -> float:
        """
        Description
        -----------
        Rotation angle in radians

        NOTE
        ----
        Restrict the rotation angle to [0-pi], therefore the tan term is 
        always positive
        """
        return np.arctan(norm(self.as_array))*2

    @staticmethod
    def rodrigues_from_quaternions(quats: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
            Vectorized batch conversion from unitary quaternions to
            Rodrigues vectors

        Parameters
        ----------
        quats: np.ndarray
            input quaternions stack along the first axis
        
        Returns
        -------
        np.ndarray
            output rodrigues vectors stack along the first axis
        """
        try:
            quats = quats.reshape(-1, 4)
        except:
            raise ValueError("Row stack input quaternions")
        
        return quats[:,1:4]/quats[:,0][:,None]


@dataclass
class Quaternion:
    """
    Unitary quaternion representation of rotation.
            q = w + x i + y j + z k
    
    reference:
        http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
    
    Note:
        No conversion methods to other representations is provided in this
        class as the conversion requires the knowledge of reference frame,
        whereas quaternion itself does not have a frame (an abstract concept).
    """
    w: float  # cos(theta/2)
    x: float  # sin(theta/2) * rotation_axis_x
    y: float  # sin(theta/2) * rotation_axis_y
    z: float  # sin(theta/2) * rotation_axis_z
    normalized: bool=False

    def __post_init__(self) -> None:
        # standardize the quaternion
        # 1. rotation angle range: [0, pi] -> self.w >= 0
        # 2. |q| === 1
        self.standardize()
    
    def standardize(self) -> None:
        _sgn = -1 if self.w < 0 else 1
        _norm = norm([self.w, self.x, self.y, self.z]) * _sgn
        self.w /= _norm
        self.x /= _norm
        self.y /= _norm
        self.z /= _norm
        self.normalized = True

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.w, self.x, self.y, self.z])
    
    @property
    def as_rodrigues(self) -> 'Rodrigues':
        _r = self.imag if iszero(self.real) else self.imag/self.real
        return Rodrigues(*_r)

    @property
    def as_eulers(self) -> 'Eulers':
        """
        Quaternion to Euler angles
        """
        # NOTE: assuming Bunge Euler angle (z->x->z)
        #   w = cos(Phi/2) * cos(phi1/2 + phi2/2)
        #   x = sin(Phi/2) * cos(phi1/2 - phi2/2)
        #   y = sin(Phi/2) * sin(phi1/2 - phi2/2)
        #   z = cos(Phi/2) * sin(phi1/2 + phi2/2)
        # ==>
        #   Phi = arcsin(sqrt(x^2+y^2)) * 2  \in [0, pi]
        # and
        #   phi1/2 - phi2/2 = arctan2(y, x)
        #   phi1/2 + phi2/2 = arctan2(z, w)
        # ==>
        #   phi1 = arctan2(z, w) + arctan2(y, x)
        #   phi2 = arctan2(z, w) - arctan2(y, x)
        
        w, x, y, z = self.as_array
        return Eulers(
            np.arctan2(z, w) + np.arctan2(y, x),
            2*np.arcsin(np.sqrt(x**2+y**2)),
            np.arctan2(z, w) - np.arctan2(y, x),
            )

    @property
    def as_matrix(self) -> np.ndarray:
        """Return the rotation matrix"""
        return self.as_eulers.as_matrix

    @property
    def real(self):
        return self.w

    @property
    def imag(self):
        return np.array([self.x, self.y, self.z])

    @property
    def norm(self) -> float:
        return np.linalg.norm(self.as_array)

    @property
    def rot_angle(self):
        return np.arccos(self.w)*2
    
    @property
    def rot_axis(self):
        return -1*normalize(self.imag) if np.arccos(self.w)<0 else normalize(self.imag)
    
    @property
    def conjugate(self) -> 'Quaternion':
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def __add__(self, other: 'Quaternion') -> 'Quaternion':
        # NOTE:
        # Adding quaternions has no physical meaning unless the results is
        # averaged to apprixmate the intermedia statem, provided that the
        # two rotations are infinitely small.
        return Quaternion(*(self.as_array + other.as_array))
    
    def __sub__(self, other: 'Quaternion') -> 'Quaternion':
        return Quaternion(*(self.as_array - other.as_array))
    
    def __neg__(self) -> 'Quaternion':
        return Quaternion(*(-self.as_array))

    @methdispatch
    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Similar to complex number multiplication
        """
        real = self.real*other.real - np.dot(self.imag, other.imag)
        imag = self.real*other.imag \
            + other.real*self.imag \
            + np.cross(self.imag, other.imag)
        return Quaternion(real, *imag)
    
    @__mul__.register(int)
    @__mul__.register(float)
    def _(self, other: Union[int, float]) -> None:
        raise ValueError("Scale a unitary quaternion is meaningless!")

    @staticmethod
    def combine_two(q2: 'Quaternion', q1: 'Quaternion') -> 'Quaternion':
        """
        Description
        -----------
        Return the quaternion that represents the compounded rotation, i.e.
            q3 = Quaternion.combine_two(q2, q1)
        where q3 is the single rotation that is equivalent to rotate by q1,
        then by q2.

        Parameters
        ----------
        q1: Quaternion
            first active rotation
        q2: Quaternion
            second active rotation

        Returns
        -------
        Quaternion
            Reduced (single-step) rotation
        """
        # NOTE:
        # Combine two operation into one is as simple as multiply them
        return q2*q1

    @staticmethod
    def average_quaternions(qs: list) -> 'Quaternion':
        """
        Description
        -----------
        Return the average quaternion based on algorithm published in
            F. Landis Markley et.al.
            Averaging Quaternions,
            doi: 10.2514/1.28949

        Parameters
        ----------
        qs: list
            list of quaternions for average
        
        Returns
        -------
        Quaternion
            average quaternion of the given list

        Note:
        This method only provides an approximation, with about 1% error. 
        > See the associated unit test for more detials.
        """
        _sum = np.sum([np.outer(q.as_array, q.as_array) for q in qs], axis=0)
        _eigval, _eigvec = np.linalg.eig(_sum/len(qs))
        return Quaternion(*np.real(_eigvec.T[_eigval.argmax()]))

    @staticmethod
    def from_angle_axis(angle: float, axis: np.ndarray) -> 'Quaternion':
        """
        Description
        -----------
        Return a unitary quaternion based on given angle and axis vector

        Parameters
        ----------
        angle: float
            rotation angle in radians (not the half angle omega)
        axis: np.ndarray
            rotation axis
        
        Retruns
        ------
        Quaternion
        """
        if iszero(angle):
            return Quaternion(1, 0, 0, 0)
        else:
            axis = normalize(axis)
            return Quaternion(np.cos(angle/2), *(np.sin(angle/2)*axis))

    @staticmethod
    def from_eulers(euler: 'Eulers') -> 'Quatrnion':
        """ Return a quaternion based on given Euler Angles """
        # allow euler as an numpy array
        # NOTE:
        #   single dispatch based polymorphysm did not work for static method
        #   therefore using try-catch block for a temp solution
        
        try:
            ee = 0.5*euler
        except:
            ee = 0.5*euler.as_array

        c1,c,c2 = np.cos(ee)
        s1,s,s2 = np.sin(ee)
        
        # NOTE: the following formular is derived from 
        #      q_euler = q_phi1*q_phi*q_phi2
        # where
        #   q_phi1 = Quaternion(np.cos(ee[0]), 0, 0, np.sin(ee[0]))  // rot_z
        #   q_phi  = Quaternion(np.cos(ee[1]), np.sin(ee[1]), 0, 0)  // rot_x
        #   q_phi2 = Quaternion(np.cos(ee[2]), 0, 0, np.sin(ee[2]))  // rot_z
        return Quaternion(
            c1*c*c2 - s1*c*s2,
            c1*s*c2 + s1*s*s2,
           -c1*s*s2 + s1*s*c2,
            c1*c*s2 + s1*c*c2,
        )

    @staticmethod
    def quaternions_from_eulers(eulers: np.ndarray) -> np.ndarray:
        """ Return a quaternion based on given Euler Angles """
        # allow euler as an numpy array
        try:
            eulers = eulers.reshape((-1, 3))
        except:
            raise ValueError(f"Eulers angles much be ROW/horizontal stacked")

        ee = 0.5*eulers
        cs = np.cos(ee)
        ss = np.sin(ee)
        c1, c, c2 = cs[:,0], cs[:,1], cs[:,2]
        s1, s, s2 = ss[:,0], ss[:,1], ss[:,2]
        
        quats = np.empty([eulers.shape[0], 4])
        quats[:,0] =  c1*c*c2 - s1*c*s2
        quats[:,1] =  c1*s*c2 + s1*s*s2
        quats[:,2] = -c1*s*s2 + s1*s*c2
        quats[:,3] =  c1*c*s2 + s1*c*c2

        return (quats/np.linalg.norm(quats, axis=1)[:,None]) * np.sign(quats[:,0])[:,None]

    @staticmethod
    def from_rodrigues(ro: 'Rodrigues') -> 'Quaternion':
        """Construct an equivalent quaternion from given Rodrigues"""
        if not isinstance(ro, Rodrigues):
            ro = Rodrigues(*ro)
        return Quaternion.from_angle_axis(ro.rot_ang, ro.rot_axis)

    @staticmethod
    def from_matrix(m: np.ndarray) -> 'Quaternion':
        """Construct quaternion from rotation matrix"""
        return Quaternion.from_eulers(Eulers.from_matrix(m))

    @staticmethod
    def from_random():
        return Quaternion.from_angle_axis(np.random.random()*np.pi, 
                                          random_three_vector()
                                        )

    @staticmethod
    def quatrotate(q: 'Quaternion', v: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Active rotate a given vector v by given unitary quaternion q
            v' = q*v_asq*q^-1

        Parameters
        ----------
        q: Quaternion
            quaternion representation of the active rotation
        v: np.ndarray
            vector

        Returns
        -------
        np.ndarray
            rotated vector
        """
        return (q.real**2 - sum(q.imag**2))*v \
            + 2*np.dot(q.imag, v)*q.imag \
            + 2*q.real*np.cross(q.imag, v)


@dataclass(frozen=True)
class Frame:
    """
    Reference frame represented as three base vectors and an origin.

    NOTE:
        Mathematically, the frame transformation should also contain scaling
        and even skewing, as the process is only related to how the base is
        defined.
        In materials science, the transformation above is not very common,
        therefore these menthods are not implemented by default.
        However, the user should still be able to extend its functionality
        either through inheritance, or simply taking advantage of the dynamic
        typing.
    """
    e1: np.ndarray = np.array([1, 0, 0])
    e2: np.ndarray = np.array([0, 1, 0])
    e3: np.ndarray = np.array([0, 0, 1])
    o:  np.ndarray = np.array([0, 0, 0])
    name: str = "lab"

    @property
    def origin(self) -> np.ndarray:
        return self.o
    
    @property
    def base(self) -> tuple:
        return (self.e1, self.e2, self.e3)

    @staticmethod
    def transformation_matrix(f1: 'Frame', f2: 'Frame') -> np.ndarray:
        """
        Description
        -----------
            Return the 3D transformation matrix (4x4) that can translate
            covariance from frame f1 to frame f2.
            
            ref:
            http://www.continuummechanics.org/coordxforms.html

        Parameters
        ----------
        f1: Frame
            original frame
        f2: Frame
            target/destination frame

        Returns
        -------
        np.ndarray
            a transformation matrix that convert the covariance in frame f1 to
            covariance in frame f2.
        """
        _m = np.zeros((4,4))
        _m[0:3, 0:3] = np.array([[np.dot(new_e, old_e) for old_e in f1.base] 
                                    for new_e in f2.base
                                ])
        _m[3,0:3] = f1.o - f2.o
        _m[3,3] = 1
        return _m

    # NOTE:
    # The following three static method provide a more general way to perform
    # rigid body manipulation of an object, including rotation and translation.
    #
    @staticmethod
    def transform_point(p_old: np.ndarray, 
                        f_old: "Frame", f_new: "Frame") -> np.ndarray:
        """
        Description
        -----------
            Transform the covariance of the given point in the old frame to
            the new frame
        
        Parameters
        ----------
        p_old: np.ndarray
            covariance of the point in the old frame (f_old)
        f_old: Frame
            old frame
        f_new: Frame
            new frame
        
        Returns
        -------
        np.ndarray
            covariance of the point in the new frame (f_new)
        """
        return np.dot(
            Frame.transformation_matrix(f_old, f_new),
            np.append(p_old, 1),
        )[:3]

    # TODO:
    # The Eienstein summation should work better here, making all the 
    # calculation essetially the same for vector and n-rank tensor.
    # Currently we are restricting everything in standard R^3.
    @staticmethod
    def transform_vector(v_old: np.ndarray,
                         f_old: "Frame", f_new: "Frame") -> np.ndarray:
        """
        Description
        -----------
            Transform the covariance of the given vector in the old frame
            f_old to the new frame f_new
        
        Parameters
        ----------
        v_old: np.ndarray
            covariance of the vector in the old frame, f_old
        f_old: Frame
            old frame
        f_new: Frame
            new frame
        
        Returns
        -------
        np.ndarray
            covariance of the vector in the new frame, f_new
        """
        return np.dot(
            Frame.transformation_matrix(f_old, f_new)[0:3, 0:3],
            v_old,
        )

    @staticmethod
    def transform_tensor(t_old: np.ndarray,
                         f_old: "Frame", f_new: "Frame") -> np.ndarray:
        """
        Description
        -----------
            Transform the covariance of the given tensor in the old frame
            f_old to the new frame f_new

        Parameters
        ----------
        t_old: np.ndarray
            covariance of the given tensor in the old frame f_old
        f_old: Frame
            old frame
        f_new: Frame
            new frame
        
        Returns
        -------
        np.ndarray
            covariance of the given tensor in the new frame f_new
        """
        _m = Frame.transformation_matrix(f_old, f_new)[0:3, 0:3]
        return np.dot(_m, np.dot(t_old, _m.T))


@dataclass
class Orientation:
    """
    Orientation is used to described a given object relative attitude with
    respect to the given reference frame, more specifically
    
        the orientation of the crystal is described as a rotation of the 
        reference frame (sample frame is a common choice) to coincide with
        the crystal’s reference frame
    
    It is worth pointing out that the pose of a rigid body contains both
    attitude and position, the description of which are both closely tied to
    the reference frame.
    """
    q: Quaternion
    f: Frame

    @property
    def frame(self) -> 'Frame':
        return self.f
    
    @frame.setter
    def frame(self, new_frame: Frame) -> None:
        # frame update
        _m = self.q.as_matrix
        for i in range(3):
            _m[:,i] = Frame.transform_vector(_m[:,i], self.frame, new_frame)
        self.q = Quaternion.from_matrix(_m)
        self.f = new_frame

    @property
    def as_quaternion(self) -> 'Quaternion':
        return self.q

    @property
    def as_rodrigues(self) -> 'Rodrigues':
        return self.q.as_rodrigues

    @property
    def as_eulers(self) -> 'Eulers':
        return self.q.as_eulers

    @property
    def as_matrix(self) -> np.ndarray:
        return self.q.as_matrix

    def misorientation(self, other: 'Orientation', lattice: str) -> tuple:
        """
        Description
        -----------
        Calculate the misorientation bewteen self and other assuming given
        lattice (symmetry)

        Parameters
        ----------
        other: Orientation
            the other orientation instance
        
        lattice: str
            symmetry name
        
        Returns
        -------
        tuple
            Return the (angle, axis) pair
        """
        # Step_1: get the symmetry operators
        sym_ops = sym_operator(lattice)
        # Step_2: make sure both are in the same frame
        if self.f.name != other.f.name:
            other.frame = self.f
        # Step_3: calculate misorientations among all possible pairs
        # NOTE:
        # 1. Quaternion multiplication q2*q1 means rotate by q1, then q2,
        #    which is why the symmetry operator is always on the right
        # 2. To calculate disorientation other -> me, we need to do the 
        #    conjudate of other to bring ? to reference frame, then from 
        #    reference frame to me, hence other.conjugate * me
        # 3. Only one symop is necessary for this process
        _drs = [other.q.conjugate * self.q * op for op in sym_ops]
        # Step_4: Locate the one pair with the smallest rotation angle
        _dr = _drs[np.argmin([me.rot_angle for me in _drs])]
        return (_dr.rot_angle, _dr.rot_axis)

    def misorientations(self, 
                        others: list, 
                        lattice: str,
                        ncores: int=2,
                    ) -> list:
        """
        Batch version of single misorientation calculation using Python native
        multi-threading library.
        """
        tmp = []
        with cf.ProcessPoolExecutor(ncores) as e:
            for other in others:
                tmp.append(e.submit(self.misorientation, other, lattice))
        return [me.result() for me in tmp]
    
    @staticmethod
    def random_orientations(n: int, frame: Frame) -> list:
        """Return n random orientations represented in the given frame"""
        # NOTE:
        # Whether this provides a uniform sampling of an orientation space
        # is not tested yet.
        return [
            Orientation(Quaternion.from_random(), frame) for _ in range(n)
        ]


def sym_operator(lattice: str) -> list:
    """
    Description
    -----------
    Return a list of symmetry operator in quaternions based on given lattice
    structure.  These quaternion are meant to operator on vectors in the
    crystal frame.

    Parameters
    ----------
    lattice: str
        lattice name

    Returns
    -------
    list
        list of quaternions as symmetry operators

    NOTE
    ----
    This function only provides a list, which is not associated with frame.
    Therefore, one need to keep in mind that these operator are meant for
    vectors in crystal frame.
    """
    if lattice is None:
        return [Quaternion(1,0,0,0)]
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


if __name__ == "__main__":

    # Example_1:
    #   reudce multi-steps active rotations (unitary quaternions) into a 
    #   single one
    from functools import reduce
    from pprint import pprint
    print("Example_1: combine multiple rotations")
    n_cases = 5
    angs = np.random.random(n_cases) * np.pi
    qs = [Quaternion.from_angle_axis(me, random_three_vector()) for me in angs]
    pprint(qs)
    print("Reduced to:")
    pprint(reduce(Quaternion.combine_two, qs))
    print()

    # Example_2:
    print("Example_2: rotate a vector")
    ang = 120
    quat = Quaternion.from_angle_axis(np.radians(ang), np.array([1,1,1]))
    vec = np.array([1,0,0])
    print(f"rotate {vec} by {quat} ({ang} deg) results in:")
    print(Quaternion.quatrotate(quat, vec))
    print()

    # Example_3:
    print("Example_3: sequential rotation is just multiplication")
    q1 = Quaternion.from_random()
    q2 = Quaternion.from_random()
    print(q1*q2)
    print(Quaternion.combine_two(q1, q2))
    print()
    # prevent the scaling of a unitary quanternion
    # q1 = q1 * 5

    # Example_4:
    print("Example_4: calc transformation matrix")
    f1 = Frame(np.array([1, 0, 0]), 
               np.array([0, 1, 0]), 
               np.array([0, 0, 1]),
               np.array([0, 0, 0]),
               'old',
            )
    sqr2 = np.sqrt(2)
    f2 = Frame(np.array([ 1/sqr2, 1/sqr2, 0]), 
               np.array([-1/sqr2, 1/sqr2, 0]),
               np.array([0, 0, 1]),
               np.array([0, 0, 0]),
               'r_z_45',
            )
    print("original frame:")
    pprint(f1)
    print("target frame:")
    pprint(f2)
    print("transformation matrix is:") 
    print(Frame.transformation_matrix(f1, f2))
    print()

    # # test quaternions_from_eulers
    # eulers = (np.random.random([5,3])-0.5) * 2 * np.pi
    # #print(eulers)
    # print(Quaternion.quaternions_from_eulers(eulers))
    # for i in range(eulers.shape[0]):
    #     e = Eulers(*(eulers[i,:]))
    #     #print(e.as_array)
    #     print(Quaternion.from_eulers(e).as_array)
