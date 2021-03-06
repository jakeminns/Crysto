import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
import math
import keras
import time
import sys
import decimal
sys.path.append(r'/home/jake/Documents/Programming/Github/Python/')
from SliceOPy import NetSlice, DataSlice
import keras.backend as K
import tensorflow as tf

def buildDensity3D(xSize,ySize,zSize,sgInfo,atomSize,sig,sg=None):
    #Select random spacegroup
    if sg == None:
        sg = np.random.randint(low=194,high=230)

    #Initilise postion table
    positionTable = []
    positionTable.append(["0",str("0"),str(0),str(0)])

    #for a random amount of atom assign a random position
    for atom in range(0,np.random.randint(1,2)):
        positionTable.append(["0",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0])])
    #positionTable = [["0","0","0","0"],["0","0.134","0.244","0.6"]]
    #build a list of symmetry operators and centering operations
    sym = sgInfo[sg][3:][0]

    cen = sgInfo[sg][4:][0]
    laue = sgInfo[sg][2][0]
    name = sgInfo[sg][1][0]
    #print(name)
    #positionTable = applyMultiPos(positionTable)
    #Apply symmetry operations
    positionTable = (buildAndApplySymmetryOpperations(positionTable,sym))
    #Apply centering operations
    positionTable = (buildAndApplySymmetryOpperations(positionTable,cen))

    #Generate empty density
    density = np.zeros((xSize,ySize,zSize))
    #Calculate modulus of x,y,z corodinates
    atomType = np.array(positionTable,dtype=np.dtype(float))[:,[0]]

    positionTable = np.array(positionTable,dtype=np.dtype(float))[:,[1,2,3]] %1
    #Round results for removal of duplicates
    positionTable = np.around(positionTable,decimals=6)
    #Remove duplicates
    positionTable = np.unique(positionTable,axis=0)
    #Add padding to account for adding atoms to boundary of box
    density = np.pad(density, atomSize, 'constant', constant_values=-1)


    for i in range(0,len(positionTable)):
        x = float(positionTable[i][0])
        y = float(positionTable[i][1])
        z = float(positionTable[i][2])

        if x<-1 or x >2 or y<-1 or y>2 or z<-1 or z>2:
            continue 
          
      # x = x % 1
      #  y= y % 1

        xi = int(np.round(x*xSize))
        yi = int(np.round(y*ySize))
        zi = int(np.round(z*zSize))

        atom = gkern3D(atomType[i],atomSize,sig)
#
        xmin = int((atomSize)+(xi-(atomSize/2)))
        xmax = int((atomSize)+(xi+(atomSize/2)))

        ymin = int((atomSize)+(yi-(atomSize/2)))
        ymax = int((atomSize)+(yi+(atomSize/2)))

        zmin = int((atomSize)+(zi-(atomSize/2)))
        zmax = int((atomSize)+(zi+(atomSize/2)))


        density[xmin:xmax,ymin:ymax,zmin:zmax] = density[xmin:xmax,ymin:ymax,zmin:zmax]  + atom
    density = density[atomSize:-atomSize,atomSize:-atomSize,atomSize:-atomSize]
    density = np.array(density)
    ff= np.fft.fftn(density).real
    
    size_ = 30
    multi = 0.3

    len_a = ff.shape[0]
    len_b = ff.shape[1]
    len_c = ff.shape[2]

    size_a = int(len_a*multi)
    size_b = int(len_b*multi)
    size_c = int(len_c*multi)

    len_a = ff.shape[0]-size_
    len_b = ff.shape[1]-size_
    len_c = ff.shape[2]-size_

    c8 = ff[0:size_,0:size_,0:size_]
    c4 = ff[0:size_,0:size_,len_c:]
    c7 = ff[0:size_,len_b:,0:size_]
    c3 = ff[0:size_,len_b:,len_c:]
    c1 = ff[len_a:,len_b:,len_c:]
    c2 = ff[len_a:,0:size_,len_c:]
    c5 = ff[len_a:,len_b:,0:size_]
    c6 = ff[len_a:,0:size_,0:size_]

    out1 = np.concatenate((c1,c2),axis=1)
    out2 = np.concatenate((c3,c4),axis=1)
    out = np.concatenate((out1,out2),axis=0)

    #out1 = np.concatenate((c5,c6),axis=1)
    #out2 = np.concatenate((c7,c8),axis=1)
    #out_top= np.concatenate((out1,out2),axis=0)
    

    #out= np.concatenate((out_bottom,out_top),axis=2)
    #out= np.concatenate((out_bottom,out_top),axis=2)

    out = np.multiply(out,out)
    #out = np.mod(out)


    out = out/np.amax(out)
    #numRot = np.random.randint(low=0,high=5)
    #listRot = [[0,1],[1,0],[0,-1],[-1,0]]
    #axis1Rot = np.random.randint(low=0,high=len(listRot))
    #outrot = np.rot90(out,k=numRot,axes=listRot[axis1Rot])
    #outrot = np.split(outrot,2)[0]

    #out = out-  np.amin(out)
    #out = out/np.amax(out)
    #out = np.multiply(out,out)
    sg1 = sg
    #out = out / np.amax(out)

    if laue=="-1":
        sg=0
    elif laue=="2/m":
        sg=1
    elif laue=="mmm":
        sg=2
    elif laue == "4/m":
        sg=3
    elif laue == "4/mmm":
        sg=4
    elif laue == "-3":
        sg = 5
    elif laue == "-3m":
        sg = 6 
    elif laue =="6/m":
        sg= 7
    elif laue == "6/mmm":
        sg = 8
    elif laue == "m-3":
        sg = 9
    elif laue == "m-3m":
        sg = 10
    else:
        sg = 11

    #print(outrot.shape)
    return density,ff,sg1-194

def buildAndApplySymmetryOpperations(positions, symmetryOpp):
    NewPositions=[]
    for sym in range(len(symmetryOpp)):
        for pos in range(len(positions)):
            tempPos =[]
            tempPos.append(positions[pos][0])
            tempPos.append(applyOpperation(positions[pos],symmetryOpp[sym][0]))
            tempPos.append(applyOpperation(positions[pos],symmetryOpp[sym][1]))
            tempPos.append(applyOpperation(positions[pos],symmetryOpp[sym][2]))
            NewPositions.append(tempPos)
    return NewPositions

def removeBrackets(input):
    
    input = str(input).replace("(","")
    input = str(input).replace(")","")
    
    return input

def applyOpperation(position, symmetryOpp):
    a = symmetryOpp.replace("x",str(removeBrackets(position[1])))
    a = a.replace("y",str(removeBrackets(position[2])))
    a = a.replace("z",str(removeBrackets(position[3])))
    return round(eval(a),3)

def readSGLib():
    file_ = open("syminfo.txt","r") #read file
    lines = file_.read().split('\n') #split file into lines 
    lines = list(filter(None, lines)) # fastest
    index = []
    sgData =[]
    for i in range(0,len(lines)):
        line = lines[i]
        if line == "begin_spacegroup":
            item = []
            sym = []
            cen = []
            laue = []
            name = []
        elif line.split()[0] == "number":
            item.append(line.split()[1])           
        elif line.split()[0] == "symop":
            sym.append(line.split()[1].split(','))
        elif line.split()[0] == "cenop":
            cen.append(line.split()[1].split(','))
        elif line.split()[0] == "symbol" and line.split()[1] == "laue":
            laue.append(line.split("'")[3])
        elif line.split()[0] == "symbol" and line.split()[1] =="xHM":
            name.append(line.split("'")[1])
        if line == "end_spacegroup":
            if item[0] not in index:
                item.append(name)
                item.append(laue)
                item.append(sym)
                item.append(cen)
                sgData.append(item) 
                index.append(item[0])

    return sgData
        
def gkern2D(kernlen=21, nsig=3):

    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    guass = fi.gaussian_filter(inp, nsig)/kernlen
    max1 = np.amax(guass)
    gauss = guass/max1
    return gauss

def lookUpAtomType(type):
    return 1.0

def gkern3D(atomType,kernlen, nsig):
    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen,kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    guass = fi.gaussian_filter(inp, nsig)/kernlen
    max1 = np.amax(guass)
    gauss = guass/max1
    density = lookUpAtomType(atomType)
    gauss = gauss * (density/np.sum(gauss))
    return gauss

def plotDensity(density):
    plt.imshow(density)
    plt.show()

def writeGRD(fdat,data,name,a,b,c,alpha,beta,gamma):

    out = open(name+".grd","w")
    out.write("Title: Put your title \n")
    cell = np.array(fdat)/10.0                                                         
    out.write(" "+str(a)+"  "+str(b)+"  "+str(c)+"  "+str(alpha)+"  "+str(beta)+"  "+str(gamma)+"\n")
    out.write("   "+str(fdat[0])+"    "+str(fdat[1])+"   "+str(fdat[2])+" \n")


    for x in range(0,(fdat[0])):
        for y in range(0,(fdat[1])):
            for z in range(0,(fdat[2])):
                out.write('%.6E' % decimal.Decimal(data[x][y][z]) + "   ")
                if z%6 == 0:
                    out.write("\n")
            out.write("\n")

def calculateSymOp(num,var,op):
    op = op.replace(var,num)
    return eval(op)

def applyMultiPos(positionTable):

    newPos = []
    for i in positionTable:

        newPos.append(i)

        for a in ["-1","+0","+1"]:
            for b in ["-1","+0","+1"]:   
                for c in ["-1","+0","+1"]:  
                    item=[]
                    item.append("0")
                    item.append(calculateSymOp(str(i[1]),"x","x"+a))  
                    item.append(calculateSymOp(str(i[2]),"y","y"+b))       
                    item.append(calculateSymOp(str(i[3]),"z","z"+c))  
                    newPos.append(item)   

    return newPos

def randomCellAngle():
    return np.random.uniform(90.5,120.0)
def randomCellParam():
    return np.random.uniform(0.5,30.0)

def randomCell(cT = None,sG=None,A=None,B=None,C=None,Alpha=None,Beta=None,Gamma=None):

    if cT == None:
        cT = np.random.randint(low=0,high=7) # Choose Cell Type 0=Tri,1=Mono,2=Orth,3=Tetr,4=Triagonal,5=Hex,6=Cubic
    
    a,b,c,alpha,beta,gamma = 0.0,0.0,0.0,0.0,0.0,0.0
    sg = 0

    if cT==0:
        a = randomCellParam()
        b = randomCellParam()
        c = randomCellParam()
        alpha = randomCellAngle()
        beta = randomCellAngle()
        gamma = randomCellAngle()
        sg = np.random.randint(low=0,high=2)
    elif cT==1:
        a = randomCellParam()
        b = randomCellParam()
        c = randomCellParam()
        alpha = 90.0
        gamma = 90.0
        beta = randomCellAngle()
        sg = np.random.randint(low=2,high=15)

    
    elif cT==2:
        a = randomCellParam()
        b = randomCellParam()
        c = randomCellParam()
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
        sg = np.random.randint(low=15,high=74)

    elif cT==3:
        a = randomCellParam()
        b = a
        c = randomCellParam()
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
        sg = np.random.randint(low=74,high=142)

    elif cT==4:
        a = randomCellParam()
        b = a
        c = randomCellParam()
        alpha = 90.0
        beta = 90.0
        gamma = 120.0
        sg = np.random.randint(low=142,high=167)

    elif cT==5:
        a = randomCellParam()
        b = a
        c = randomCellParam()
        gamma = 120.0
        alpha = 90.0
        beta = 90.0
        sg = np.random.randint(low=167,high=194)

    elif cT==6:
        a = randomCellParam()
        b = a
        c = a
        gamma = 90.0
        alpha = 90.0
        beta = 90.0
        sg = np.random.randint(low=194,high=230)

    else:       
        print("Incorrect Cell Type Chosen")
        sys.exit()   

    if sG != None:
        sg = sG
    
    if A != None:
        a = A
    
    if  B != None:
        b = B

    if C != None:
        c = C

    if Alpha != None:
        alpha = Alpha
    if Beta != None:
        beta = Beta
    if Gamma != None:
        gamma = Gamma

    return a,b,c,alpha,beta,gamma,sg

def convertToPosition(positionTable,a,b,c,alpha,beta,gamma):
    
    #positionTable = np.array(positionTable)
    rad = np.pi/180.0
    omega = a*b*c*(1-(np.cos(alpha*rad)**2)-(np.cos(beta*rad)**2)-(np.cos(gamma*rad)**2)+2*np.cos(alpha*rad)*np.cos(beta*rad)*np.cos(gamma*rad))**0.5
    convertArr = np.array([[a,b*np.cos(gamma*rad),c*np.cos(beta*rad)],[0,b*np.sin(gamma*rad),c*((np.cos(beta*rad)*np.cos(gamma*rad)-np.cos(gamma*rad))/(np.sin(gamma*rad)))],[0.0,0.0,omega/(a*b*np.sin(gamma*rad))]])
    out = []
    for i in positionTable:
        out.append(np.matmul(np.array(i).transpose(),convertArr).tolist())

    return out

def buildSF3D(hShape,kShape,lShape,sgInfo,atomSize,sig):
    #Select random spacegroup

    a,b,c,alpha,beta,gamma,sg = randomCell()
    #Initilise postion table
    positionTable = []
    #positionTable.append(["0",str("0"),str(0),str(0)])

    #for a random amount of atom assign a random position
    for atom in range(0,np.random.randint(1,4)):
        positionTable.append(["0",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0])])
   # positionTable = [["0","0","0","0"],["0","0.123","0.321","0.213"]]
    #build a list of symmetry operators and centering operations
    sym = sgInfo[sg][3:][0]

    cen = sgInfo[sg][4:][0]
    laue = sgInfo[sg][2][0]
    name = sgInfo[sg][1][0]

    #print(name)
    
    #positionTable = applyMultiPos(positionTable)
    #Apply symmetry operations
    positionTable = (buildAndApplySymmetryOpperations(positionTable,sym))
    #Apply centering operations
    positionTable = (buildAndApplySymmetryOpperations(positionTable,cen))

    positionTable = np.array(positionTable,dtype=np.dtype(float))[:,[1,2,3]] %1
    #Round results for removal of duplicates
    positionTable = np.around(positionTable,decimals=6)
    #Remove duplicates
    positionTable = np.unique(positionTable,axis=0)
    #positions = np.array(positionTable,dtype=np.dtype(float))[:,[1,2,3]]
    #positionTable = convertToPosition(positionTable,a,b,c,alpha,beta,gamma)

    #Generate empty density
   # ff = np.zeros((hShape[1]-hShape[0],kShape[1]-kShape[0],lShape[1]-lShape[0]))

   # for h in range(0,np.abs(hShape[1])+np.abs(hShape[0])):
   #     for k in range(0,np.abs(kShape[1])+np.abs(kShape[0])):
   #         for l in range(0,np.abs(lShape[1])+np.abs(lShape[0])):
   #             for atom in positionTable:
   #                 ff[h][k][l] = ff[h][k][l] + (1.0 * np.exp(-2j*np.pi*((h+hShape[0])*atom[0]+(k+kShape[0])*atom[1]+(l+hShape[0])*atom[2])))
    #

    h = np.arange(hShape[0],hShape[1])
    k = np.arange(kShape[0],kShape[1])
    l = np.arange(lShape[0],lShape[1])
    #h.shape[0]*k.shape[0]*l.shape[0]
    grid = np.array(np.meshgrid(h,k,l)).T.reshape(h.shape[0]*k.shape[0]*l.shape[0],3,order="C")
    #Generate empty density
    ff2 = np.full((len(positionTable),h.shape[0]*k.shape[0]*l.shape[0],1), 0.+0.j)
    VFunc = np.vectorize(SF)

    for atom in range(0,len(positionTable)):
        temp = VFunc(grid,positionTable[atom])
        t1 = np.multiply(temp[:,[0]],temp[:,[1]],casting='same_kind')
        ff2[atom] = np.multiply(t1,temp[:,[2]],casting='same_kind')
        #ff2[atom] = np.prod(temp,axis=1)
    #
    ff2 = np.sum(ff2,axis=0).reshape(h.shape[0],k.shape[0],l.shape[0],order="F")
    #ff2 = np.flip(ff2,0)
    ff2 = np.flip(ff2,2)


    #print(a,b,c,alpha,beta,gamma,sg)

    #ff2r = ff2.real
    #ff2r = np.fft.fftn(ff2).real
    
    ff2 = np.multiply(ff2.real,ff2.real)
    ff2 = ff2/np.amax(ff2)
    #ff1r = np.fft.ifftn(ff).real

    #ff = np.multiply(ff,ff)
    #ff = ff/np.amax(ff)

    return ff2,sg

def SF(ml,atom):
    return (np.exp(-2j*np.pi*(ml*atom)))

def vecSF(positionTable):
        print(positionTable)
        h = np.arange(-15,15)
        k = np.arange(-15,15)
        l = np.arange(-15,15)
        #h.shape[0]*k.shape[0]*l.shape[0]
        grid = np.array(np.meshgrid(h,k,l)).T.reshape(h.shape[0]*k.shape[0]*l.shape[0],3,order="C")
        VFunc = np.vectorize(SF)

        temp = VFunc(grid,positionTable)
        t1 = np.multiply(temp[:,[0]],temp[:,[1]],casting='same_kind')
        return np.multiply(t1,temp[:,[2]],casting='same_kind')
        #ff2[atom] = np.prod(temp,axis=1)


def genrateTrainingData(num,funcParams):

    dataFeat = []
    dataLabel = []

    for i in range(0,num):

        if i % 100 == 0:
            print(i)
        item = buildSF3D(*funcParams)
        dataFeat.append(item[0])
        dataLabel.append(item[1])

    dataFeat = np.array(dataFeat)
    dataLabel = np.array(dataLabel)

    np.save("feat.npy",dataFeat)
    np.save("label.npy",dataLabel)


def buildModelConv(input_shape,num_classes):
    
    model = keras.Sequential()
    model.add(keras.layers.Conv3D(12, (3,3,3),input_shape=input_shape,padding="same",data_format= keras.backend.image_data_format()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPooling3D(pool_size=(2, 2,2),data_format= keras.backend.image_data_format()))
#    
    model.add(keras.layers.Conv3D(12, (3,3, 3),padding="same",data_format= keras.backend.image_data_format()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.MaxPooling3D(pool_size=(2, 2,2),data_format= keras.backend.image_data_format()))
   # model.add(keras.layers.Dropout(0.25))
##    
    #model.add(keras.layers.Conv3D(64, (3,3,3),data_format= K.image_data_format()))
    #model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Activation('relu'))
    #777model.add(keras.layers.Conv2D(64,(3,3),data_format=K.image_data_format()))
    #model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.MaxPooling3D(pool_size=(2,2,2),data_format=keras.backend.image_data_format()))
    model.add(keras.layers.Dropout(0.35))

    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.AveragePooling3D(pool_size=(2,2 ,2),data_format= K.image_data_format()))
    #model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    #model.add(keras.layers.Dense(1048))
    #model.add(keras.layers.BatchNormalization())
    #model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.Dense(456))
    #model.add(keras.layers.Activation('relu'))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(36))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('sigmoid'))

    return model

def buildModelDense(input_shape,num_classes):
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(keras.layers.Dense(35000,input_dim=13500,activation='sigmoid'))
    model.add(keras.layers.Dense(12000,activation='sigmoid'))
    model.add(keras.layers.Dense(1000,activation='sigmoid'))
    model.add(keras.layers.Dense(59,activation='sigmoid'))

    
    return model
