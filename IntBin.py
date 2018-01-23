import numpy as np
import struct

"""
Note:
1. Can only integrate to one degree
2. This function just put all data together, so there may be some pixel show multiple times in the data chunk. e.g. same (x,y) value appears multiple times.
3. Due to the same reason, the Peak IDs will be messed up (Peak ID from several images are put together to one image)
4. Although issues 2 and 3 exist, I9 still works fine with integerated binary files. Look at "ImageData.cpp".
"""
output='IntS13/S13_Int_z1_'
ImagePar={'nDetectors':3,
        'sBinFilePrefix':'/home/fyshen13/Dec_16/S13z1Bin/Ti7_S13_Reduced_z1_',
        'nReductionNSUM':4,
        'nBinFileIndexStart':0,
        'fOmegaStart':-90,
        'fOmegaStop':90,
        }

def IntegrateBinFiles(oPar,outputprefix):
    """
    Note: only can integrate to one degree
    """
    nDegrees=oPar['fOmegaStop']-oPar['fOmegaStart']
    for k in range(oPar['nDetectors']):
        indx=oPar['nBinFileIndexStart']
        remap_indx=indx
        for i in range(nDegrees):
            integ=[[],[],[],[]]
            for j in range(oPar['nReductionNSUM']):
                print 'Reading:',oPar['sBinFilePrefix']+"{0:06d}".format(indx)+'.bin'+str(k)
                try:
                    bi=ReadI9BinaryFiles(oPar['sBinFilePrefix']+"{0:06d}".format(indx)+'.bin'+str(k))
                except:
                    print 'Reading failed'
                for ii in range(4):
                    integ[ii].extend(bi[ii])
                indx=indx+1
            print 'Writing:',outputprefix+'{0:06d}'.format(remap_indx)+'.bin'+str(k)
            filename=outputprefix+'{0:06d}'.format(remap_indx)+'.bin'+str(k)
            try:
                WritePeakBinaryFile(integ,filename)
            except:
                print 'Writing failed'
            remap_indx=remap_indx+1

def ReadI9BinaryFiles(filename):
    """
    %%           [float(32) version header]
    %%           [SBlockHeader]
    %%           [SubHeader for first coordinate]
    %%           [ n uint(16) pixel first coordinate ]
    %%           [SubHeader for second coordinate]
    %%           [ n uint(16) pixel second coordinate ]
    %%           [SubHeader for intensity ]
    %%           [ n float(32) pixel intensity ]
    %%           [SubHeader for peakID ]
    %%           [ n uint(32) for peakID ]
    """
    fid=open(filename,'rb')
    headerMain={}
    fid.read(4)
    ReadUFFHeader(fid,headerMain)
#    print headerMain['NameSize']
    ChildXLoc=struct.unpack('I',fid.read(4))[0]
    ChildYLoc=struct.unpack('I',fid.read(4))[0]
    ChildIntLoc=struct.unpack('I',fid.read(4))[0]
    CildPeakLoc=struct.unpack('I',fid.read(4))[0]
    NumPeaks=struct.unpack('I',fid.read(4))[0]

    Header1={}
    ReadUFFHeader(fid,Header1)
#    print Header1['NameSize']
    nElements=(Header1['DataSize']-Header1['NameSize'])/2
    x=struct.unpack('{0:d}H'.format(nElements),fid.read(2*nElements))

    Header1={}
    ReadUFFHeader(fid,Header1)
    nCheck=(Header1['DataSize']-Header1['NameSize'])/2
#    print Header1['NameSize']
    if nCheck!=nElements:
        raise Exception('Number of elements mismatch')
    y=struct.unpack('{0:d}H'.format(nElements),fid.read(2*nElements))


    Header1={}
    ReadUFFHeader(fid,Header1)
    nCheck=(Header1['DataSize']-Header1['NameSize'])/4
#    print Header1['NameSize']
    if nCheck!=nElements:
        raise Exception('Number of elements mismatch')
    intensity=struct.unpack('{0:d}f'.format(nElements),fid.read(4*nElements))


    Header1={}
    ReadUFFHeader(fid,Header1)
    nCheck=(Header1['DataSize']-Header1['NameSize'])/2
#    print Header1['NameSize']
    if nCheck!=nElements:
        raise Exception('Number of elements mismatch')
    PeakID=struct.unpack('{0:d}H'.format(nElements),fid.read(2*nElements))

    fid.close()
    return np.array(x),np.array(y),np.array(intensity),np.array(PeakID)

def ReadUFFHeader(fid,header):
    tmp=fid.read(4)
    uBlockHeader=struct.unpack('I',tmp)[0]
    if uBlockHeader != int('FEEDBEEF',16):
        raise Exception('file is corrupted')
    header['BlockType']=struct.unpack('H',fid.read(2))[0]
    header['DataFormat']=struct.unpack('H',fid.read(2))[0]
    header['NumChildren']=struct.unpack('H',fid.read(2))[0]
    header['NameSize']=struct.unpack('H',fid.read(2))[0]
    header['DataSize']=struct.unpack('I',fid.read(4))[0]
    header['ChunkNumber']=struct.unpack('H',fid.read(2))[0]
    header['TotalChunks']=struct.unpack('H',fid.read(2))[0]
    header['BlockName']=struct.unpack('{0:d}s'.format(header['NameSize']),fid.read(header['NameSize']))[0]

def WritePeakBinaryFile(snp,sFilename):
    headerMain={'BlockType':1,'DataFormat':1,'NumChildren':4,'NameSize':9,'BlockName':'PeakFile','DataSize':0,'ChunkNumber':0,'TotalChunks':0}

    headerX={'BlockType':1,'DataFormat':1,'NumChildren':0,'NameSize':12,'BlockName':'PixelCoord0','DataSize':0,'ChunkNumber':0,'TotalChunks':0}
    headerY={'BlockType':1,'DataFormat':1,'NumChildren':0,'NameSize':12,'BlockName':'PixelCoord1','DataSize':0,'ChunkNumber':0,'TotalChunks':0}
    headerInt={'BlockType':1,'DataFormat':1,'NumChildren':0,'NameSize':10,'BlockName':'Intensity','DataSize':0,'ChunkNumber':0,'TotalChunks':0}
    headerPeakID={'BlockType':1,'DataFormat':1,'NumChildren':0,'NameSize':7,'BlockName':'PeakID','DataSize':0,'ChunkNumber':0,'TotalChunks':0}
    n=len(snp[0])
    headerPeakID['DataSize']=n*2+7
    headerInt['DataSize']=n*4+10
    headerX['DataSize']=n*2+12
    headerY['DataSize']=n*2+12
    ChildrenPtrSize=4*4
    HeaderSize=20
    ChildXLoc=4
    ChildYLoc=ChildXLoc+headerX['DataSize']+HeaderSize
    ChildIntLoc=ChildYLoc+headerY['DataSize']+HeaderSize
    ChildPeakLoc=ChildIntLoc+headerInt['DataSize']+HeaderSize
    headerMain['DataSize']=ChildPeakLoc+headerPeakID['DataSize']+HeaderSize+ChildrenPtrSize+4

    f=open(sFilename,'wb')
    f.write(struct.pack('f',1))
    WriteUFFHeader(f,headerMain)
    f.write(struct.pack('I',ChildXLoc))
    f.write(struct.pack('I',ChildYLoc))
    f.write(struct.pack('I',ChildIntLoc))
    f.write(struct.pack('I',ChildPeakLoc))

    if n>0:
        nPeak=len(np.unique(snp[3]))
    else:
        nPeak=0
    f.write(struct.pack('I',nPeak))

    WriteUFFHeader(f,headerX)
    if n>0:
        f.write(struct.pack('{0:d}H'.format(n),*snp[0]))
    WriteUFFHeader(f,headerY)
    if n>0:
        f.write(struct.pack('{0:d}H'.format(n),*snp[1]))
    WriteUFFHeader(f,headerInt)

    if n>0:
        f.write(struct.pack('{0:d}f'.format(n),*snp[2]))
    WriteUFFHeader(f,headerPeakID)
    if n>0:
        f.write(struct.pack('{0:d}H'.format(n),*snp[3]))
    f.close()

def WriteUFFHeader(fid,header):
    uBlockHeader = int('FEEDBEEF',16)
    fid.write(struct.pack('I',uBlockHeader))
    fid.write(struct.pack('H',header['BlockType']))
    fid.write(struct.pack('H',header['DataFormat']))
    fid.write(struct.pack('H',header['NumChildren']))
    fid.write(struct.pack('H',header['NameSize']))
    fid.write(struct.pack('I',header['DataSize']))
    fid.write(struct.pack('H',header['ChunkNumber']))
    fid.write(struct.pack('H',header['TotalChunks']))
    fid.write(struct.pack('{0:d}s'.format(header['NameSize']),header['BlockName']))

def main():
    IntegrateBinFiles(ImagePar,output)
    return

if __name__=='__main__': main()


