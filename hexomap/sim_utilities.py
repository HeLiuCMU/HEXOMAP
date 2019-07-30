# this part copy from Yufeng Shen's Code:
#https://github.com/Yufeng-shen/nfHEDMtools/blob/master/Simulation.py
import numpy as np
from fractions import Fraction
from math import floor
from hexomap import utility
# from matplotlib import path




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
        print("Norm: ", self.Norm)
        print("CoordOrigin: ", self.CoordOrigin)
        print("J vector: ", self.Jvector)
        print("K vector: ", self.Kvector)


class CrystalStr:
    def __init__(self, material='new'):
        self.name = material
        self.AtomPos = []
        self.AtomZs = []
        self.symtype = None
        if material == 'gold':
            self.symtype = 'Cubic'
            self.PrimA = 4.08 * np.array([1, 0, 0])
            self.PrimB = 4.08 * np.array([0, 1, 0])
            self.PrimC = 4.08 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 79)
            self.addAtom([0, 0.5, 0.5], 79)
            self.addAtom([0.5, 0, 0.5], 79)
            self.addAtom([0.5, 0.5, 0], 79)
        elif material == 'copper':
            self.symtype = 'Cubic'
            self.PrimA = 3.61 * np.array([1, 0, 0])
            self.PrimB = 3.61 * np.array([0, 1, 0])
            self.PrimC = 3.61 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 29)
            self.addAtom([0, 0.5, 0.5], 29)
            self.addAtom([0.5, 0, 0.5], 29)
            self.addAtom([0.5, 0.5, 0], 29)
        elif material == 'copperBCC':
            self.symtype = 'Cubic'
            self.PrimA = 2.947 * np.array([1, 0, 0])
            self.PrimB = 2.947 * np.array([0, 1, 0])
            self.PrimC = 2.947 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 29)
            self.addAtom([0.5, 0.5, 0.5], 29)
        elif material == 'copperFCC':
            self.symtype = 'Cubic'
            self.PrimA = 3.692 * np.array([1, 0, 0])
            self.PrimB = 3.692 * np.array([0, 1, 0])
            self.PrimC = 3.692 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 29)
            self.addAtom([0, 0.5, 0.5], 29)
            self.addAtom([0.5, 0, 0.5], 29)
            self.addAtom([0.5, 0.5, 0], 29)
        elif material == 'stainless_steel':
            self.symtype = 'Cubic'
            self.PrimA = 3.59 * np.array([1, 0, 0])
            self.PrimB = 3.59 * np.array([0, 1, 0])
            self.PrimC = 3.59 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 26)
            self.addAtom([0, 0.5, 0.5], 26)
            self.addAtom([0.5, 0, 0.5], 26)
            self.addAtom([0.5, 0.5, 0], 26)
        elif material == 'iron_bcc':
            # bcc lattice
            self.symtype = 'Cubic'
            self.PrimA = 2.856 * np.array([1, 0, 0])
            self.PrimB = 2.856 * np.array([0, 1, 0])
            self.PrimC = 2.856 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 26)
            self.addAtom([0.5, 0.5, 0.5], 26)
        elif material == 'iron_fcc':
            self.symtype = 'Cubic'
            self.PrimA = 2.856 * np.array([1, 0, 0])
            self.PrimB = 2.856 * np.array([0, 1, 0])
            self.PrimC = 2.856 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 26)
            self.addAtom([0, 0.5, 0.5], 26)
            self.addAtom([0.5, 0, 0.5], 26)
            self.addAtom([0.5, 0.5, 0], 26)
        elif material == 'SrTiO3':
            self.symtype = 'Cubic'
            self.PrimA = 3.9053 * np.array([1, 0, 0])
            self.PrimB = 3.9053 * np.array([0, 1, 0])
            self.PrimC = 3.9053 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 22)
            self.addAtom([0.5, 0.5, 0.5], 38)
            self.addAtom([0.5, 0, 0], 8)
            self.addAtom([0, 0.5, 0], 8)
            self.addAtom([0, 0, 0.5], 8)
        elif material == 'SrTiO3_v1':
            self.symtype = 'Cubic'
            self.PrimA = 3.9053 * np.array([1, 0, 0])
            self.PrimB = 3.9053 * np.array([0, 1, 0])
            self.PrimC = 3.9053 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 38)
            self.addAtom([0.5, 0.5, 0.5], 22)
            self.addAtom([0.5, 0.5, 0], 8)
            self.addAtom([0, 0.5, 0.5], 8)
            self.addAtom([0.5, 0, 0.5], 8)
        elif material == 'SrTiO3_v2':
            self.symtype = 'Cubic'
            self.PrimA = 3.9053 * np.array([1, 0, 0])
            self.PrimB = 3.9053 * np.array([0, 1, 0])
            self.PrimC = 3.9053 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 38)
            #self.addAtom([0.5, 0.5, 0.5], 22)
            self.addAtom([0.5, 0.5, 0], 8)
            self.addAtom([0, 0.5, 0.5], 8)
            self.addAtom([0.5, 0, 0.5], 8)
        elif material == 'SrTiO3_v3':
            self.symtype = 'Cubic'
            self.PrimA = 3.9053 * np.array([1, 0, 0])
            self.PrimB = 3.9053 * np.array([0, 1, 0])
            self.PrimC = 3.9053 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 38)
            #self.addAtom([0.5, 0.5, 0.5], 22)
            self.addAtom([0.5, 0.5, 0], 38)
            self.addAtom([0, 0.5, 0.5], 38)
            self.addAtom([0.5, 0, 0.5], 38)

        
        elif material == 'Ti7':
            self.symtype = 'Hexagonal'
            self.PrimA = 2.92539 * np.array([1, 0, 0])
            self.PrimB = 2.92539 * np.array([np.cos(np.pi * 2 / 3), np.sin(np.pi * 2 / 3), 0])
            self.PrimC = 4.67399 * np.array([0, 0, 1])
            self.addAtom([1 / 3.0, 2 / 3.0, 1 / 4.0], 22)
            self.addAtom([2 / 3.0, 1 / 3.0, 3 / 4.0], 22)
        elif material == 'WE43':
            # not tested, use Mg to approximate
            self.symtype = 'Hexagonal'
            a = 3.2094
            c = 5.2107 
            self.PrimA = a * np.array([1, 0, 0])
            self.PrimB = a * np.array([np.cos(np.pi * 2 / 3), np.sin(np.pi * 2 / 3), 0])
            self.PrimC = c * np.array([0, 0, 1])
            self.addAtom([1 / 3.0, 2 / 3.0, 1 / 4.0], 12)
            self.addAtom([2 / 3.0, 1 / 3.0, 3 / 4.0], 12)
        elif material == 'Ti64_alpha':
            self.symtype = 'Hexagonal'
            self.PrimA = 2.930 * np.array([1, 0, 0])
            self.PrimB = 2.930 * np.array([np.cos(np.pi * 2 / 3), np.sin(np.pi * 2 / 3), 0])
            self.PrimC = 4.677 * np.array([0, 0, 1])
            self.addAtom([1 / 3.0, 2 / 3.0, 1 / 4.0], 22)
            self.addAtom([2 / 3.0, 1 / 3.0, 3 / 4.0], 22)
        elif material == 'Ti64_beta':
            # bcc lattice
            self.symtype = 'Cubic'
            self.PrimA = 3.224 * np.array([1, 0, 0])
            self.PrimB = 3.224 * np.array([0, 1, 0])
            self.PrimC = 3.224 * np.array([0, 0, 1])
            self.addAtom([0, 0, 0], 26)
            self.addAtom([0.5, 0.5, 0.5], 26)
        elif material == 'UO2':
            # bcc lattice
            self.symtype = 'Cubic'
            self.PrimA = 5.471 * np.array([1, 0, 0])
            self.PrimB = 5.471 * np.array([0, 1, 0])
            self.PrimC = 5.471 * np.array([0, 0, 1])  
            self.addAtom([0, 0, 0], 92)
        elif material.lower() in ['zr', ' zirconium']:
            # hexagonal lattice
            # unit: angstrom, radian
            # source:
            # https://www.webelements.com/zirconium/crystal_structure.html
            self.symtype = 'Hexagonal'
            self.PrimA = 3.232 * np.array([1, 0, 0])
            self.PrimB = 3.232 * np.array([np.cos(np.pi * 2 / 3), np.sin(np.pi * 2 / 3), 0])
            self.PrimC = 5.147 * np.array([0, 0, 1])
            self.addAtom([1 / 3.0, 2 / 3.0, 1 / 4.0], 22)
            self.addAtom([2 / 3.0, 1 / 3.0, 3 / 4.0], 22)
        elif material.endswith(('.yml', '.yaml')):
            d = utility.load_yaml(material)
            self.symtype = d['symtype']
            if d['symtype'] == 'Hexagonal':
                self.PrimA = d['PrimA'] * np.array([1, 0, 0])
                self.PrimB = d['PrimB'] * np.array([np.cos(np.pi * 2 / 3), np.sin(np.pi * 2 / 3), 0])
                self.PrimC = d['PrimC'] * np.array([0, 0, 1])
            elif d['symtype'] == 'Cubic':
                self.PrimA = d['PrimA'] * np.array([1, 0, 0])
                self.PrimB = d['PrimB'] * np.array([0, 1, 0])
                self.PrimC = d['PrimC'] * np.array([0, 0, 1]) 
            else:
                raise NotImplementedError('symType should be Cubic or Hexagonal')
            for key, value in d['Atom'].items():
                self.addAtom(value['pos'], value['atomNumber'])
        else:
            raise ValueError("Unknown mateiral type!")

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

def main():

    m0 = CrystalStr('../examples/material_example/hexagonal_Zr.yml')
    m1 = CrystalStr('zr')
    d0 = m0.__dict__
    d1 = m1.__dict__
    d0.pop('name')
    d1.pop('name')
    print("====================== hexagonal ====================")
    print(d0)
    print(d1)
    m0 = CrystalStr('../examples/material_example/simpleCubic_UO2.yml')
    m1 = CrystalStr('UO2')
    d0 = m0.__dict__
    d1 = m1.__dict__
    d0.pop('name')
    d1.pop('name')
    print("====================== simpleCubic ====================")
    print(d0)
    print(d1)
    m0 = CrystalStr('../examples/material_example/cubic_iron_bcc.yml')
    m1 = CrystalStr('iron_bcc')
    d0 = m0.__dict__
    d1 = m1.__dict__
    d0.pop('name')
    d1.pop('name')
    print("====================== bcc ====================")
    print(d0)
    print(d1)
    m0 = CrystalStr('../examples/material_example/cubic_iron_fcc.yml')
    m1 = CrystalStr('iron_fcc')
    d0 = m0.__dict__
    d1 = m1.__dict__
    d0.pop('name')
    d1.pop('name')
    print("====================== fcc ====================")
    print(d0)
    print(d1)
if __name__ == "__main__":
    # execute only if run as a script
    main()
