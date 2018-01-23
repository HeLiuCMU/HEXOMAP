# this part copy from Yufeng Shen's Code:
#https://github.com/Yufeng-shen/nfHEDMtools/blob/master/Simulation.py
import numpy as np
from fractions import Fraction
from math import floor
from matplotlib import path


def frankie_angles_from_g(g, verbo=True, **exp):
    """
    Converted from David's code, which converted from Bob's code.
    I9 internal simulation coordinates: x ray direction is positive x direction, positive z direction is upward, y direction can be determined by right hand rule.
    I9 mic file coordinates: x, y directions are the same as the simulation coordinates.
    I9 detector images (include bin and ascii files): J is the same as y, K is the opposite z direction.
    The omega is along positive z direction.

    Parameters
    ------------
    g: array
       One recipropcal vector in the sample frame when omega==0. Unit is ANGSTROM^-1.
    exp:
        Experimental parameters. If use 'wavelength', the unit is 10^-10 meter; if use 'energy', the unit is keV.

    Returns
    -------------
    2Theta and eta are in radian, chi, omega_a and omega_b are in degree. omega_a corresponding to positive y direction scatter, omega_b is negative y direction scatter.
    """
    ghat = g / np.linalg.norm(g);
    if 'wavelength' in exp:
        sin_theta = np.linalg.norm(g) * exp['wavelength'] / (4 * np.pi);
    elif 'energy' in exp:
        sin_theta = np.linalg.norm(g) / (exp['energy'] * 0.506773182) / 2
    cos_theta = np.sqrt(1 - sin_theta ** 2);
    cos_chi = ghat[2];
    sin_chi = np.sqrt(1 - cos_chi ** 2);
    omega_0 = np.arctan2(ghat[0], ghat[1]);

    if np.fabs(sin_theta) <= np.fabs(sin_chi):
        phi = np.arccos(sin_theta / sin_chi);
        sin_phi = np.sin(phi);
        eta = np.arcsin(sin_chi * sin_phi / cos_theta);
        delta_omega = np.arctan2(ghat[0], ghat[1]);
        delta_omega_b1 = np.arcsin(sin_theta / sin_chi);
        delta_omega_b2 = np.pi - delta_omega_b1;
        omega_res1 = delta_omega + delta_omega_b1;
        omega_res2 = delta_omega + delta_omega_b2;
        if omega_res1 > np.pi:
            omega_res1 -= 2 * np.pi;
        if omega_res1 < -np.pi:
            omega_res1 += 2 * np.pi;
        if omega_res2 > np.pi:
            omega_res2 -= 2 * np.pi;
        if omega_res2 < -np.pi:
            omega_res2 += 2 * np.pi;
    else:
        return -1
    if verbo == True:
        print '2theta: ', 2 * np.arcsin(sin_theta) * 180 / np.pi
        print 'chi: ', np.arccos(cos_chi) * 180 / np.pi
        print 'phi: ', phi * 180 / np.pi
        print 'omega_0: ', omega_0 * 180 / np.pi
        print 'omega_a: ', omega_res1 * 180 / np.pi
        print 'omega_b: ', omega_res2 * 180 / np.pi
        print 'eta: ', eta * 180 / np.pi
    return {'chi': np.arccos(cos_chi) * 180 / np.pi, '2Theta': 2 * np.arcsin(sin_theta), 'eta': eta,
            'omega_a': omega_res1 * 180 / np.pi, 'omega_b': omega_res2 * 180 / np.pi, 'omega_0': omega_0 * 180 / np.pi}


class Detector:
    def __init__(self):
        self.Norm = np.array([0, 0, 1])
        self.CoordOrigin = np.array([0., 0., 0.])
        self.Jvector = np.array([1, 0, 0])
        self.Kvector = np.array([0, -1, 0])
        self.PixelJ = 0.00148
        self.PixelK = 0.00148
        self.NPixelJ = 2048
        self.NPixelK = 2048

    def Move(self, J, K, trans, tilt):
        self.CoordOrigin -= J * self.Jvector * self.PixelJ + K * self.Kvector * self.PixelK
        self.CoordOrigin = tilt.dot(self.CoordOrigin) + trans
        self.Norm = tilt.dot(self.Norm)
        self.Jvector = tilt.dot(self.Jvector)
        self.Kvector = tilt.dot(self.Kvector)

    def IntersectionIdx(self, ScatterSrc, TwoTheta, eta, bIdx=True):
        #print('eta:{0}'.format(eta))
        #self.Print()
        dist = self.Norm.dot(self.CoordOrigin - ScatterSrc)
        scatterdir = np.array([np.cos(TwoTheta), np.sin(TwoTheta) * np.sin(eta), np.sin(TwoTheta) * np.cos(eta)])
        InterPos = dist / (self.Norm.dot(scatterdir)) * scatterdir + ScatterSrc
        J = (self.Jvector.dot(InterPos - self.CoordOrigin) / self.PixelJ)
        K = (self.Kvector.dot(InterPos - self.CoordOrigin) / self.PixelK)
        if 0 <= int(J) < self.NPixelJ and 0 <= int(K) < self.NPixelK:
            if bIdx == True:
                return int(J), int(K)
            else:
                return J, K
        else:
            return -1

    def BackProj(self, HitPos, omega, TwoTheta, eta):
        """
        HitPos: ndarray (3,)
                The position of hitted point on lab coord, unit in mm
        """
        scatterdir = np.array([np.cos(TwoTheta), np.sin(TwoTheta) * np.sin(eta), np.sin(TwoTheta) * np.cos(eta)])
        t = HitPos[2] / (np.sin(TwoTheta) * np.cos(eta))
        x = HitPos[0] - t * np.cos(TwoTheta)
        y = HitPos[1] - t * np.sin(TwoTheta) * np.sin(eta)
        truex = np.cos(omega) * x + np.sin(omega) * y
        truey = -np.sin(omega) * x + np.cos(omega) * y
        return np.array([truex, truey])

    def Idx2LabCord(self, J, K):
        return J * self.PixelJ * self.Jvector + K * self.PixelK * self.Kvector + self.CoordOrigin

    def Reset(self):
        self.__init__()

    def Print(self):
        print "Norm: ", self.Norm
        print "CoordOrigin: ", self.CoordOrigin
        print "J vector: ", self.Jvector
        print "K vector: ", self.Kvector


class CrystalStr:
    def __init__(self, material='new'):
        self.AtomPos = []
        self.AtomZs = []
        if material == 'gold':
            self.PrimA = 4.08 * np.array([1, 0, 0])
            self.PrimB = 4.08 * np.array([0, 1, 0])
            self.PrimC = 4.08 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 79)
            self.addAtom([0, 0.5, 0.5], 79)
            self.addAtom([0.5, 0, 0.5], 79)
            self.addAtom([0.5, 0.5, 0], 79)
        elif material == 'copper':
            self.PrimA = 3.61 * np.array([1, 0, 0])
            self.PrimB = 3.61 * np.array([0, 1, 0])
            self.PrimC = 3.61 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 29)
            self.addAtom([0, 0.5, 0.5], 29)
            self.addAtom([0.5, 0, 0.5], 29)
            self.addAtom([0.5, 0.5, 0], 29)
        elif material == 'Ti7':
            self.PrimA = 2.92539 * np.array([1, 0, 0])
            self.PrimB = 2.92539 * np.array([np.cos(np.pi * 2 / 3), np.sin(np.pi * 2 / 3), 0])
            self.PrimC = 4.67399 * np.array([0, 0, 1])
            self.addAtom([1 / 3.0, 2 / 3.0, 1 / 4.0], 22)
            self.addAtom([2 / 3.0, 1 / 3.0, 3 / 4.0], 22)
        else:
            pass

    def setPrim(self, x, y, z):
        self.PrimA = np.array(x)
        self.PrimB = np.array(y)
        self.PrimC = np.array(z)

    def addAtom(self, pos, Z):
        self.AtomPos.append(np.array(pos))
        self.AtomZs.append(Z)

    def getRecipVec(self):
        self.RecipA = 2 * np.pi * np.cross(self.PrimB, self.PrimC) / (self.PrimA.dot(np.cross(self.PrimB, self.PrimC)))
        self.RecipB = 2 * np.pi * np.cross(self.PrimC, self.PrimA) / (self.PrimB.dot(np.cross(self.PrimC, self.PrimA)))
        self.RecipC = 2 * np.pi * np.cross(self.PrimA, self.PrimB) / (self.PrimC.dot(np.cross(self.PrimA, self.PrimB)))

    def calStructFactor(self, hkl):
        F = 0
        for ii in range(len(self.AtomZs)):
            F += self.AtomZs[ii] * np.exp(-2 * np.pi * 1j * (hkl.dot(self.AtomPos[ii])))
        return F

    def getGs(self, maxQ):
        self.Gs = []
        maxh = int(maxQ / float(np.linalg.norm(self.RecipA)))
        maxk = int(maxQ / float(np.linalg.norm(self.RecipB)))
        maxl = int(maxQ / float(np.linalg.norm(self.RecipC)))
        for h in range(-maxh, maxh + 1):
            for k in range(-maxk, maxk + 1):
                for l in range(-maxl, maxl + 1):
                    if h == 0 and k == 0 and l == 0:
                        pass
                    else:
                        G = h * self.RecipA + k * self.RecipB + l * self.RecipC
                        if np.linalg.norm(G) <= maxQ:
                            if np.absolute(self.calStructFactor(np.array([h, k, l]))) > 1e-6:
                                self.Gs.append(G)
        self.Gs = np.array(self.Gs)


def GetProjectedVertex(Det1, sample, orien, etalimit, grainpos, getPeaksInfo=False, bIdx=True, omegaL=-90, omegaU=90,
                       **exp):
    """
    Get the observable projected vertex on a single detector and their G vectors.
    Caution!!! This function only works for traditional nf-HEDM experiment setup.

    Parameters
    ------------
    Det1: Detector
            Remember to move this detector object to correct position first.
    sample: CrystalStr
            Must calculated G list
    orien:  ndarray
            Active rotation matrix of orientation at that vertex
    etalimit: scalar
            Limit of eta value. Usually is about 85.
    grainpos: array
            Position of that vertex in mic file, unit is mm.
    exp: dict
        X ray energy or wavelength

    Returns
    ------------
    Peaks: ndarray
            N*3 ndarray, records position of each peak. The first column is the J value, second is K value, third is omega value in degree.
    Gs: ndarray
        N*3 ndarray, records  corresponding G vector in sample frame.
    """
    Peaks = []
    Gs = []
    PeaksInfo = []
    rotatedG = orien.dot(sample.Gs.T).T
    for g1 in rotatedG:
        res = frankie_angles_from_g(g1, verbo=False, **exp)
        if res == -1:
            pass
        elif res['chi'] >= 90:
            pass
        elif res['eta'] > etalimit:
            pass
        else:
            if omegaL <= res['omega_a'] <= omegaU:
                omega = res['omega_a'] / 180.0 * np.pi
                newgrainx = np.cos(omega) * grainpos[0] - np.sin(omega) * grainpos[1]
                newgrainy = np.cos(omega) * grainpos[1] + np.sin(omega) * grainpos[0]
                idx = Det1.IntersectionIdx(np.array([newgrainx, newgrainy, 0]), res['2Theta'], res['eta'], bIdx)
                if idx != -1:
                    Peaks.append([idx[0], idx[1], res['omega_a']])
                    Gs.append(g1)
                    if getPeaksInfo:
                        PeaksInfo.append({'WhichOmega': 'a', 'chi': res['chi'], 'omega_0': res['omega_0'],
                                          '2Theta': res['2Theta'], 'eta': res['eta']})
            if omegaL <= res['omega_b'] <= omegaU:
                omega = res['omega_b'] / 180.0 * np.pi
                newgrainx = np.cos(omega) * grainpos[0] - np.sin(omega) * grainpos[1]
                newgrainy = np.cos(omega) * grainpos[1] + np.sin(omega) * grainpos[0]
                idx = Det1.IntersectionIdx(np.array([newgrainx, newgrainy, 0]), res['2Theta'], -res['eta'], bIdx)
                if idx != -1:
                    Peaks.append([idx[0], idx[1], res['omega_b']])
                    Gs.append(g1)
                    if getPeaksInfo:
                        PeaksInfo.append({'WhichOmega': 'b', 'chi': res['chi'], 'omega_0': res['omega_0'],
                                          '2Theta': res['2Theta'], 'eta': -res['eta']})
    Peaks = np.array(Peaks)
    Gs = np.array(Gs)
    if getPeaksInfo:
        return Peaks, Gs, PeaksInfo
    return Peaks, Gs


def digitize(xy):
    """
    xy: ndarray shape(4,2)
        J and K indices in float, four points. This digitize method is far from ideal

    Returns
    -------------
    f: list
        list of integer tuples (J,K) that is hitted. (filled polygon)

    """
    p = path.Path(xy)

    def line(pixels, x0, y0, x1, y1):
        if x0 == x1 and y0 == y1:
            pixels.append((x0, y0))
            return
        brev = True
        if abs(y1 - y0) <= abs(x1 - x0):
            x0, y0, x1, y1 = y0, x0, y1, x1
            brev = False
        if x1 < x0:
            x0, y0, x1, y1 = x1, y1, x0, y0
        leny = abs(y1 - y0)
        for i in range(leny + 1):
            if brev:
                pixels.append(
                    tuple((int(round(Fraction(i, leny) * (x1 - x0))) + x0, int(1 if y1 > y0 else -1) * i + y0)))
            else:
                pixels.append(
                    tuple((int(1 if y1 > y0 else -1) * i + y0, int(round(Fraction(i, leny) * (x1 - x0))) + x0)))

    bnd = p.get_extents().get_points().astype(int)
    ixy = xy.astype(int)
    pixels = []
    line(pixels, ixy[0, 0], ixy[0, 1], ixy[1, 0], ixy[1, 1])
    line(pixels, ixy[1, 0], ixy[1, 1], ixy[2, 0], ixy[2, 1])
    line(pixels, ixy[2, 0], ixy[2, 1], ixy[3, 0], ixy[3, 1])
    line(pixels, ixy[3, 0], ixy[3, 1], ixy[0, 0], ixy[0, 1])
    points = []
    for jj in range(bnd[0, 0], bnd[1, 0] + 1):
        for kk in range(bnd[0, 1], bnd[1, 1] + 1):
            points.append((jj, kk))
    points = np.asarray(points)
    mask = p.contains_points(points)

    ipoints = points[mask]

    f = list([tuple(ii) for ii in ipoints])
    f.extend(pixels)

    return f