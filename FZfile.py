# Generate finer FZ file
# He Liu
# CMU
# 20180122
# see reference paper: http://refbase.cvc.uab.es/files/PIE2012.pdf
import numpy as np
import RotRep
def  generate_random_rot_mat(nEuler):
    '''
    generate random euler angles, and bring back to FZ
    :param nEuler: number of Euler angles
    :param symtype: 'Cubic' or 'Hexagonal'
    :return: array, nEulerx3
    '''
    # ############### method 1 ###############
    # not vectorized
    nEuler = int(nEuler)
    alpha = np.random.uniform(-np.pi,np.pi,nEuler)
    gamma = np.random.uniform(-np.pi,np.pi,nEuler)
    z = np.random.uniform(-1,1,nEuler)
    beta = np.arccos(z)
    result = np.empty([nEuler,3,3])
    for i in range(nEuler):
        matTmp = RotRep.EulerZXZ2Mat(np.array([alpha[i],beta[i],gamma[i]]))
        result[i,:,:] = matTmp
    return result
    ########## vectorized version ##############
def write_mat_to_file(mat,fName):
    '''
    write rotation matrix to file
    :param mat:
    :param fName:
    :return:
    '''
    np.savetxt(fName,mat)
def test_mat_to_euler():
    e = np.array([10.1237, 75.4599, 340.791])/180*np.pi
    mat = RotRep.EulerZXZ2Mat(e)
    print(mat.shape)
    eResult = np.array(RotRep.Mat2EulerZXZ(mat))/np.pi*180
    print(eResult)

def random_angle_around_mat(mat,nAngle,boundBox,symtype):
    '''
    !!!!!CAUTION!!! current version ignore crystal type, does not bring back to FZ.
    generate rotation angles around certain rotation matrix,
    :param mat: input rotation matrix, [n_mat,3,3]
    :param nAngle: number of angles to generate
    :param boundBox: in radian, angle range to generate [-boundBox,boundBox]
    :param symtype: 'Cubic' or 'Hexagonal'
    :return: [n_mat*nAngle,3,3] matrix
    test passed
    '''
    ################### NOT Vectorized #######################################
    # mat = mat.reshape([-1,3,3])
    # #print(mat.shape)
    # result = np.empty([mat.shape[0]*nAngle, 3,3])
    # #print(result.shape)
    # for i in range(mat.shape[0]):
    #     eulerTmp = RotRep.Mat2EulerZXZ(mat[i,:,:])
    #     alpha = np.random.uniform(eulerTmp[0]-boundBox, eulerTmp[0]+boundBox, nAngle)
    #     gamma = np.random.uniform(eulerTmp[2]-boundBox, eulerTmp[2]+boundBox, nAngle)
    #     z = np.random.uniform(np.cos(eulerTmp[1])-boundBox*np.sin(eulerTmp[1]),np.cos(eulerTmp[1])+boundBox*np.sin(eulerTmp[1]), nAngle) # cos(a+b) ~ cosa+b*sina
    #     z[z>1] = 1
    #     z[z<-1] = -1
    #     beta = np.arccos(z)
    #     for j in range(nAngle):
    #         matTmp = RotRep.EulerZXZ2Mat(np.array([alpha[j], beta[j], gamma[j]]))
    #         #result[i*mat.shape[0]+j, :,:] = np.array(RotRep.Orien2FZ(matTmp, symtype=symtype)[0])
    #         result[i * mat.shape[0] + j, :, :] = matTmp
    #     result[i*mat.shape[0],:, : ] = mat[i,:,:]
    # return result
    ################## vectorized version 0.1###############################3
    # mat = mat.reshape([-1,3,3])
    # result = np.empty([mat.shape[0] * nAngle, 3, 3])
    # #print(result.shape)
    # eulerTmp = RotRep.Mat2EulerZXZVectorized(mat)
    # eulerTmp = eulerTmp.repeat(nAngle, axis=0)
    # eulerTmp[:, 0] = eulerTmp[:, 0] + np.random.uniform(-boundBox,+boundBox,eulerTmp.shape[0])
    # eulerTmp[:, 2] = eulerTmp[:, 2] + np.random.uniform(-boundBox, +boundBox, eulerTmp.shape[0])
    # z = np.empty(mat.shape[0]*nAngle)
    # for i in range(mat.shape[0]):
    #     z[i*mat.shape[0]:i*mat.shape[0]+nAngle] = np.random.uniform(np.cos(eulerTmp[i*mat.shape[0],1])-boundBox*np.sin(eulerTmp[i*mat.shape[0],1]),
    #                                                                 np.cos(eulerTmp[i*mat.shape[0],1])+boundBox*np.sin(eulerTmp[i*mat.shape[0],1]),
    #                                                                 nAngle)
    # z[z>1] = 1
    # z[z<-1] = -1
    # beta = np.arccos(z)
    # result = RotRep.EulerZXZ2MatVectorized(eulerTmp)
    # result[np.arange(mat.shape[0])*nAngle,:,:] = mat
    # return result
    ################### vectorized version 0.2 ###############
    ################### NOT Vectorized #######################################
    mat = mat.reshape([-1,3,3])
    #print(mat.shape)
    result = np.empty([mat.shape[0]*nAngle, 3,3])
    #print(result.shape)
    eulerTmp = RotRep.Mat2EulerZXZVectorized(mat)
    randEulerTmp = np.empty([nAngle,3])
    for i in range(mat.shape[0]):
        randEulerTmp[:,0] = np.random.uniform(eulerTmp[i,0]-boundBox, eulerTmp[i,0]+boundBox, nAngle)
        randEulerTmp[:,2] = np.random.uniform(eulerTmp[i,2]-boundBox, eulerTmp[i,2]+boundBox, nAngle)
        z = np.random.uniform(np.cos(eulerTmp[i,1])-boundBox*np.sin(eulerTmp[i,1]),np.cos(eulerTmp[i,1])+boundBox*np.sin(eulerTmp[i,1]), nAngle) # cos(a+b) ~ cosa+b*sina
        z[z>1] = 1
        z[z<-1] = -1
        randEulerTmp[:,1] = np.arccos(z)
        result[i * mat.shape[0]: (i*mat.shape[0] + nAngle),:,:] = RotRep.EulerZXZ2MatVectorized(randEulerTmp)
        result[i*mat.shape[0],:, : ] = mat[i,:,:]
    return result

if __name__ =='__main__':
    #mat = generate_random_rot_mat(50000,'Hexagonal')
    #print(mat)
    #write_mat_to_file(mat,'FZ_MAT.txt')
    #test_mat_to_euler()
    euler = np.array([174.956, 55.8283, 182.94])/180*np.pi
    print(euler)
    mat = RotRep.EulerZXZ2Mat(euler)
    #print(mat)
    result = random_angle_around_mat(mat,10,0.1,'Hexagonal')
    print(result.shape)
    #print(result)
    for i in range(result.shape[0]):
        print(RotRep.Mat2EulerZXZ(result[i,:,:]))
        print(RotRep.Misorien2FZ1(mat,result[i,:,:],symtype='Hexagonal'))

    print (RotRep.Mat2EulerZXZ(mat))