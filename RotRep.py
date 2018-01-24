import matplotlib.pyplot as plt
import numpy as np
from math import atan2


def rod_from_quaternion(quat):
    '''
    adapt from:
    function rod = RodOfQuat(quat)
        % RodOfQuat - Rodrigues parameterization from quaternion.
        %
        %   USAGE:
        %
        %   rod = RodOfQuat(quat)
        %
        %   INPUT:
        %
        %   quat is 4 x n,
        %        an array whose columns are quaternion paramters;
        %        it is assumed that there are no binary rotations
        %        (through 180 degrees) represented in the input list
        %
        %   OUTPUT:
        %
        %  rod is 3 x n,
        %      an array whose columns form the Rodrigues parameterization
        %      of the same rotations as quat
        %
        rod = quat(2:4, :)./repmat(quat(1,:), [3 1]);

    :param quat:
    :return:
    '''
    if quat.ndim == 1:
        rod = quat[1:4] / quat[0]
    else:
        rod = quat[1:4, :] / np.repeat(np.expand_dims(quat[0, :], axis=0), 3, axis=0)
    return rod


def quaternion_from_matrix(matrix, isprecise=False):
    """"

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def Q2Mat(q0, q1, q2, q3):
    """
    convert active quarternion to active matrix
    """
    m = np.matrix([[1 - 2 * q2 ** 2 - 2 * q3 ** 2, 2 * q1 * q2 + 2 * q0 * q3, 2 * q1 * q3 - 2 * q0 * q2],
                   [2 * q1 * q2 - 2 * q0 * q3, 1 - 2 * q1 ** 2 - 2 * q3 ** 2, 2 * q2 * q3 + 2 * q0 * q1],
                   [2 * q1 * q3 + 2 * q0 * q2, 2 * q2 * q3 - 2 * q0 * q1, 1 - 2 * q1 ** 2 - 2 * q2 ** 2]])
    return m


def Euler2Mat(e):
    """
    Active Euler Angle (radian)  in ZYZ convention to active rotation matrix, which means newV=M*oldV
    """
    x = e[0]
    y = e[1]
    z = e[2]
    s1 = np.sin(x)
    s2 = np.sin(y)
    s3 = np.sin(z)
    c1 = np.cos(x)
    c2 = np.cos(y)
    c3 = np.cos(z)
    m = np.array([[c1 * c2 * c3 - s1 * s3, -c3 * s1 - c1 * c2 * s3, c1 * s2],
                  [c1 * s3 + c2 * c3 * s1, c1 * c3 - c2 * s1 * s3, s1 * s2],
                  [-c3 * s2, s2 * s3, c2]])
    return m


def EulerZXZ2Mat(e):
    """
    Active Euler Angle (radian)  in ZXZ convention to active rotation matrix, which means newV=M*oldV
    """
    x = e[0]
    y = e[1]
    z = e[2]
    s1 = np.sin(x)
    s2 = np.sin(y)
    s3 = np.sin(z)
    c1 = np.cos(x)
    c2 = np.cos(y)
    c3 = np.cos(z)
    m = np.array([[c1 * c3 - c2 * s1 * s3, -c1 * s3 - c3 * c2 * s1, s1 * s2],
                  [s1 * c3 + c2 * c1 * s3, c1 * c2 * c3 - s1 * s3, -c1 * s2],
                  [s3 * s2, s2 * c3, c2]])
    return m

def EulerZXZ2MatVectorized(e):
    '''
    He Liu
    20180124
    vectorized verison, convert multiple euler the same time
    for 10000 angles:
    EulerZXZ2Mat time: 0.22568488121
    EulerZXZ2MatVectorized time: 0.0045428276062


    :param e: [n_euler,3]matrix
    :return: [n_euler,3,3] rotation matrix
    '''
    e = e.reshape([-1,3])
    m = np.empty([e.shape[0],3,3])
    x = e[:,0]
    y = e[:,1]
    z = e[:,2]
    s1 = np.sin(x)
    s2 = np.sin(y)
    s3 = np.sin(z)
    c1 = np.cos(x)
    c2 = np.cos(y)
    c3 = np.cos(z)
    m[:,0,0] = c1 * c3 - c2 * s1 * s3
    m[:,0,1] = -c1 * s3 - c3 * c2 * s1
    m[:,0,2] =  s1 * s2
    m[:,1,0] = s1 * c3 + c2 * c1 * s3
    m[:,1,1] = c1 * c2 * c3 - s1 * s3
    m[:,1,2] = -c1 * s2
    m[:,2,0] = s3 * s2
    m[:,2,1] = s2 * c3
    m[:,2,2] = c2
    return m
def GetSymRotMat(symtype='Cubic'):
    """
    return an array of active rotation matrices of the input crystal symmetry

    Parameters
    ----------
    symtype: string
            Symmetry type of crystal. For now only 'Cubic' and 'Hexagonal' are implemented.

    Returns
    ----------
    m:  ndarray
        A three dimensional numpy array, which has the shape (n,3,3).
    """
    if symtype == 'Cubic':
        m = np.zeros((24, 3, 3))
        m[0][0, 1] = 1
        m[0][1, 0] = -1
        m[0][2, 2] = 1

        m[1][0, 0] = -1
        m[1][1, 1] = -1
        m[1][2, 2] = 1

        m[2][0, 1] = -1
        m[2][1, 0] = 1
        m[2][2, 2] = 1

        m[3][0, 2] = -1
        m[3][1, 1] = 1
        m[3][2, 0] = 1

        m[4][0, 0] = -1
        m[4][1, 1] = 1
        m[4][2, 2] = -1

        m[5][0, 2] = 1
        m[5][1, 1] = 1
        m[5][2, 0] = -1

        m[6][0, 0] = 1
        m[6][1, 2] = 1
        m[6][2, 1] = -1

        m[7][0, 0] = 1
        m[7][1, 1] = -1
        m[7][2, 2] = -1

        m[8][0, 0] = 1
        m[8][1, 2] = -1
        m[8][2, 1] = 1

        m[9][0, 1] = 1
        m[9][1, 2] = 1
        m[9][2, 0] = 1

        m[10][0, 2] = 1
        m[10][1, 0] = 1
        m[10][2, 1] = 1

        m[11][0, 2] = -1
        m[11][1, 0] = 1
        m[11][2, 1] = -1

        m[12][0, 1] = 1
        m[12][1, 2] = -1
        m[12][2, 0] = -1

        m[13][0, 2] = 1
        m[13][1, 0] = -1
        m[13][2, 1] = -1

        m[14][0, 1] = -1
        m[14][1, 2] = -1
        m[14][2, 0] = 1

        m[15][0, 2] = -1
        m[15][1, 0] = -1
        m[15][2, 1] = 1

        m[16][0, 1] = -1
        m[16][1, 2] = 1
        m[16][2, 0] = -1

        m[17][0, 0] = -1
        m[17][1, 2] = 1
        m[17][2, 1] = 1

        m[18][0, 2] = 1
        m[18][1, 1] = -1
        m[18][2, 0] = 1

        m[19][0, 1] = 1
        m[19][1, 0] = 1
        m[19][2, 2] = -1

        m[20][0, 0] = -1
        m[20][1, 2] = -1
        m[20][2, 1] = -1

        m[21][0, 2] = -1
        m[21][1, 1] = -1
        m[21][2, 0] = -1

        m[22][0, 1] = -1
        m[22][1, 0] = -1
        m[22][2, 2] = -1

        m[23][0, 0] = 1
        m[23][1, 1] = 1
        m[23][2, 2] = 1

        return m
    elif symtype == 'Hexagonal':
        m = np.zeros((12, 3, 3))
        m[0][0, 0] = 0.5
        m[0][1, 1] = 0.5
        m[0][2, 2] = 1
        m[0][0, 1] = -np.sqrt(3) * 0.5
        m[0][1, 0] = np.sqrt(3) * 0.5

        m[1] = m[0].dot(m[0])
        m[2] = m[1].dot(m[0])
        m[3] = m[2].dot(m[0])
        m[4] = m[3].dot(m[0])
        m[5] = np.eye(3)

        m[6][0, 0] = 1
        m[6][1, 1] = -1
        m[6][2, 2] = -1

        m[7] = m[0].dot(m[6])
        m[8] = m[1].dot(m[6])
        m[9] = m[2].dot(m[6])
        m[10] = m[3].dot(m[6])
        m[11] = m[4].dot(m[6])

        return m
    else:
        print "not implemented yet"
        return 0


def Orien2FZ(m, symtype='Cubic'):
    """
    Reduce orientation to fundamental zone, input and output are both active matrices
    Careful, it is m*op not op*m

    Parameters
    -----------
    m:      ndarray
            Matrix representation of orientation
    symtype:string
            The crystal symmetry

    Returns
    -----------
    oRes:   ndarray
            The rotation matrix after reduced. Note that this function doesn't actually
            reduce the orientation to fundamental zone, only make sure the angle is the
            smallest one, so there are multiple orientations have the same angle but
            different directions. oRes is only one of them.
    angle:  scalar
            The reduced angle.
    """
    ops = GetSymRotMat(symtype)
    angle = 6.3
    for op in ops:
        #print(op)
        tmp = m.dot(op)
        cosangle = 0.5 * (tmp.trace() - 1)
        cosangle = min(0.9999999, cosangle)
        cosangle = max(-0.9999999, cosangle)
        newangle = np.arccos(cosangle)
        if newangle < angle:
            angle = newangle
            oRes = tmp
    return oRes, angle


# def plane2FZ(v,symtype='Cubic'):
#    V=v.reshape((3,1))
#    if symtype=='Cubic':
#        ops=GetSymRotMat(symtype)
#        for op in ops:
#            oRes=op.dot(V)
#            if oRes[0]>oRes[1] and oRes[0]>oRes[2] and oRes[1]>0 and oRes[2]>0:
#                break
#        return oRes

def Misorien2FZ1(m1, m2, symtype='Cubic'):
    """
    Careful, it is m1*op*m2T, the misorientation in sample frame, the order matters. Only returns the angle, doesn't calculate the right axis direction

    Parameters
    -----------
    m1:     ndarray
            Matrix representation of orientation1
    m2:     ndarray
            Matrix representation of orientation2
    symtype:string
            The crystal symmetry

    Returns
    -----------
    oRes:   ndarray
            The misorientation matrix after reduced. Note that this function doesn't actually
            reduce the orientation to fundamental zone, only make sure the angle is the
            smallest one, so there are multiple orientations have the same angle but
            different directions. oRes is only one of them.
    angle:  scalar
            The misorientation angle.
    """
    m2 = np.matrix(m2)
    ops = GetSymRotMat(symtype)
    angle = 6.3
    for op in ops:
        tmp = m1.dot(op.dot(m2.T))
        cosangle = 0.5 * (tmp.trace() - 1)
        cosangle = min(0.9999999999, cosangle)
        cosangle = max(-0.99999999999, cosangle)
        newangle = np.arccos(cosangle)
        if newangle < angle:
            angle = newangle
            oRes = tmp
    return oRes, angle


def Misorien2FZ2(m1, m2, symtype='Cubic'):
    """
    Careful, we need misorientation in crystal frame (eg. m2), it should be o2*m2T*m1*o1, the order matters. Then change m1 and m2 (just do transpose).

    Parameters
    -----------
    m1:     ndarray
            Matrix representation of orientation1
    m2:     ndarray
            Matrix representation of orientation2
    symtype:string
            The crystal symmetry

    Returns
    -----------
    axis:   ndarray
            The unit vector of rotation direction.
    angle:  scalar
            The misorientation angle. (0~180 degree)
    """
    if symtype != 'Cubic':
        print "only calculate axis for cubic symmetry"
        return
    m2 = np.matrix(m2)
    dm = (m2.T).dot(m1)
    ops = GetSymRotMat(symtype)
    angle = 6.3
    for op1 in ops:
        for op2 in ops:
            tmp = op2.dot(dm.dot(op1))
            cosangle = 0.5 * (tmp.trace() - 1)
            cosangle = min(0.9999999, cosangle)
            cosangle = max(-0.9999999, cosangle)
            newangle = np.arccos(cosangle)
            if newangle < angle:
                sina = np.sin(newangle)
                direction = np.zeros(3)
                direction[0] = (tmp[2, 1] - tmp[1, 2]) / 2.0 / sina
                direction[1] = (tmp[0, 2] - tmp[2, 0]) / 2.0 / sina
                direction[2] = (tmp[1, 0] - tmp[0, 1]) / 2.0 / sina
                if direction[0] > direction[1] and direction[1] > direction[2] and direction[2] > 0:
                    angle = newangle
                    axis = direction
                else:
                    direction = -direction
                    if direction[0] > direction[1] and direction[1] > direction[2] and direction[2] > 0:
                        angle = newangle
                        axis = direction

    return axis, angle


def Misorien2FZ3(m1, m2, symtype='Cubic'):
    """
    Careful, we need misorientation in crystal frame (eg. m2), it should be o2*m2T*m1*o1, the order matters. Then change m1 and m2 (just do transpose).

    Parameters
    -----------
    m1:     ndarray
            Matrix representation of orientation1
    m2:     ndarray
            Matrix representation of orientation2
    symtype:string
            The crystal symmetry

    Returns
    -----------
    axis:   ndarray
            The unit vector of rotation direction.
    angle:  scalar
            The misorientation angle. (0~180 degree)
    """
    if symtype != 'Cubic':
        print "only calculate axis for cubic symmetry"
        return
    m2 = np.matrix(m2)
    dm = (m2.T).dot(m1)
    ops = GetSymRotMat(symtype)
    angle = 6.3
    for op1 in ops:
        for op2 in ops:
            tmp = op2.dot(dm.dot(op1))
            cosangle = 0.5 * (tmp.trace() - 1)
            cosangle = min(0.9999999, cosangle)
            cosangle = max(-0.9999999, cosangle)
            newangle = np.arccos(cosangle)
            if newangle < angle:
                w, W = np.linalg.eig(tmp)
                i = np.where(abs(np.real(w) - 1) < 1e-8)[0]
                direction = np.asarray(np.real(W[:, i[-1]])).squeeze()
                if abs(direction[0]) > 1e-8:
                    sina = (tmp[2, 1] - tmp[1, 2]) / 2.0 / direction[0]
                    if sina < 0:
                        direction = -direction
                    if direction[0] > direction[1] and direction[1] > direction[2] and direction[2] > 0:
                        angle = newangle
                        axis = direction
                tmp = tmp.T
                w, W = np.linalg.eig(tmp)
                i = np.where(abs(np.real(w) - 1) < 1e-8)[0]
                direction = np.asarray(np.real(W[:, i[-1]])).squeeze()
                if abs(direction[0]) > 1e-8:
                    sina = (tmp[2, 1] - tmp[1, 2]) / 2.0 / direction[0]
                    if sina < 0:
                        direction = -direction
                    if direction[0] > direction[1] and direction[1] > direction[2] and direction[2] > 0:
                        angle = newangle
                        axis = direction

    return axis, angle


def Mat2Euler(m):
    """
    transform active rotation matrix to euler angles in ZYZ convention
    """
    threshold = 0.9999999
    if m[2, 2] > threshold:
        x = 0
        y = 0
        z = atan2(m[1, 0], m[0, 0])
    elif m[2, 2] < -threshold:
        x = 0
        y = np.pi
        z = atan2(m[0, 1], m[0, 0])
    else:
        x = atan2(m[1, 2], m[0, 2])
        y = atan2(np.sqrt(m[2, 0] ** 2 + m[2, 1] ** 2), m[2, 2])
        #        y=np.arccos(m[2,2])
        z = atan2(m[2, 1], -m[2, 0])
    # if np.sin(x)*m[1,2]<0 or np.cos(x)*m[0,2]<0 : x=x+np.pi
    #    if np.sin(z)*m[2,1]<0 or np.cos(z)*m[2,0]>0 : z=z+np.pi
    if x < 0: x = x + 2 * np.pi
    if y < 0: y = y + 2 * np.pi
    if z < 0: z = z + 2 * np.pi
    return x, y, z


def Mat2EulerZXZ(m):
    """
    transform active rotation matrix to euler angles in ZXZ convention, not right(seems right now)
    """
    threshold = 0.9999999
    if m[2, 2] > threshold:
        x = 0
        y = 0
        z = atan2(m[1, 0], m[0, 0])
    elif m[2, 2] < -threshold:
        x = 0
        y = np.pi
        z = atan2(m[0, 1], m[0, 0])
    else:
        x = atan2(m[0, 2], -m[1, 2])
        y = atan2(np.sqrt(m[2, 0] ** 2 + m[2, 1] ** 2), m[2, 2])
        z = atan2(m[2, 0], m[2, 1])
    if x < 0: x = x + 2 * np.pi
    if y < 0: y = y + 2 * np.pi
    if z < 0: z = z + 2 * np.pi
    return x, y, z
def Mat2EulerZXZVectorized(m):
    '''
    he liu
    vectorized verion of Mat2EulerZXZ
    compute 10000 angles time difference:
    Mat2EulerZXZ: 0.0466799736023
    Mat2EulerZXZVectorized   0.0028178691864

    :param m: [n_mat,3,3] array
    :return: [n_mat,3] array, euler angles
    '''
    threshold = 0.9999999
    euler = np.empty([m.shape[0],3])
    idx0 = m[:,2,2] > threshold
    idx1 = m[:,2,2] < -threshold
    idx2 = np.bitwise_and(m[:,2,2] < threshold,m[:,2,2] > -threshold)
    #print(m[idx0,0,0])
    euler[idx0, 0] = 0
    euler[idx0, 1] = 0
    euler[idx0, 2] = np.arctan2(m[idx0, 1, 0], m[idx0, 0, 0])
    euler[idx1, 0] = 0
    euler[idx1, 1] = np.pi
    euler[idx1, 2] = np.arctan2(m[idx1, 0, 1], m[idx1, 0, 0])
    euler[idx2, 0] = np.arctan2(m[idx2, 0, 2], -m[idx2, 1, 2])
    euler[idx2, 1] = np.arctan2(np.sqrt(m[idx2, 2, 0] ** 2 + m[idx2, 2, 1] ** 2), m[idx2, 2, 2])
    euler[idx2, 2] = np.arctan2(m[idx2, 2, 0], m[idx2, 2, 1])
    euler[euler[:, 0] < 0, 0] = euler[euler[:, 0] < 0, 0]+2 * np.pi
    euler[euler[:, 1] < 0, 1] = euler[euler[:, 1] < 0, 1] + 2 * np.pi
    euler[euler[:, 2] < 0, 2] = euler[euler[:, 2] < 0, 2] + 2 * np.pi
    return euler
    # def plot(m,symtype='Cubic'):
    #    ops=GetSymRotMat(symtype)
    #    for op in ops:
    #        tmp=op.dot(m)
    #        x,y,z=Mat2Euler(tmp)
    #        plt.scatter(x,y+z)
    #    plt.show()
    # def plot2(m,symtype='Cubic'):
    #    ops=GetSymRotMat(symtype)
    #    for op1 in ops:
    #        for op2 in ops:
    #            tmp=op1.dot(m.dot(op2))
    #            x,y,z=Mat2Euler(tmp)
    #            plt.scatter(x,y+z)
    #    plt.show()

def benchmark_e2m():
    # benchmark speed of vectorized version and not
    nEuler = 10000
    alpha = np.random.uniform(-np.pi,np.pi,nEuler)
    gamma = np.random.uniform(-np.pi,np.pi,nEuler)
    z = np.random.uniform(-1,1,nEuler)
    beta = np.arccos(z)
    euler = np.concatenate([alpha[:,np.newaxis],beta[:,np.newaxis],gamma[:,np.newaxis]],axis=1)
    import time

    start = time.time()
    for i in range(euler.shape[0]):
        m = EulerZXZ2Mat(euler[i,:])
    end = time.time()
    print('EulerZXZ2Mat time: {0}'.format(end - start))
    print(m)
    start = time.time()
    m = EulerZXZ2MatVectorized(euler)
    end = time.time()
    print('EulerZXZ2MatVectorized time: {0}'.format(end - start))
    print(m[-1,:,:])
def benchmark_m2e():
    # benchmark speed of vectorized version m2e
    import FZfile
    import time
    m = FZfile.generate_random_rot_mat(10000)
    start = time.time()
    for i in range(m.shape[0]):
        e = Mat2EulerZXZ(m[i,:,:])
    end = time.time()
    print(end-start)
    print(e)
    start = time.time()
    e = Mat2EulerZXZVectorized(m)
    end = time.time()
    print(end-start)
    print(e[-1,:])
if __name__ =='__main__':
    benchmark_m2e()
    #Orien2FZ(np.zeros([3,3]))
 #    m1 = np.array([[ 0.94245757,  0.26655785,  0.20179359],
 # [ 0.11294816,  0.31423632, -0.94260185],
 # [-0.31466879,  0.91115446,  0.26604718]])
 #    m2 = np.array([[ 0.99736739, -0.00341931,  0.07243343],
 # [-0.05797104,  0.56247327,  0.82478069],
 # [-0.04356205, -0.8268084,   0.56079427]])
 #    print(Misorien2FZ1(Orien2FZ(m2,'Hexagonal')[0],m1,'Hexagonal'))