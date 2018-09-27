import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as fi
import math
import keras
import sys
import decimal
sys.path.append(r'/home/minns_jake/downloads/')
from SliceOPy import NetSlice, DataSlice
import keras.backend as K


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
    print(len(lines))
    lines = list(filter(None, lines)) # fastest
    index = []
    sgData =[]
    for i in range(0,len(lines)):
        line = lines[i]
        if line == "begin_spacegroup":
            item = []
            sym = []
            cen = []
        if line.split()[0] == "number":
            item.append(line.split()[1])           
        if line.split()[0] == "symop":
            sym.append(line.split()[1].split(','))
        if line.split()[0] == "cenop":
            cen.append(line.split()[1].split(','))  
        if line == "end_spacegroup":
            if item[0] not in index:
                item.append(sym)
                item.append(cen)
                sgData.append(item) 
                index.append(item[0])
   


    return sgData
        

def readSGInfo():
    file_ = open("spgra.dat","r") #read file
    lines = file_.read().split('\n') #split file into lines
    info = [] #info list init
    item = [] #item list init
    prev = False #blank prev variable to track if the first item in the previous line was a number 
    for line in lines: #loop lines
        if len(line) > 0: #check line isn't blank
            if is_number(line.split()[0]): #check if the first item in the line is a number
                if prev == False: #if the previous line was NOT a number we know this is the start of a new space group item so append the previous item to the info and initiize a new blank list 
                    info.append(item)
                    item = []

                prev = True #this line start with a number
            else:
                prev = False #if the line doesn't start with a number set this to false
            if item == []: #if the item is blank we know that line is the line containing the spacegroup number, we want to keep that
                item.append(line.split()[0])

            elif is_number(line.split()[0]) ==False: #if the first item in the line is not a number we know it is a symmetry operator so add that
                item.append(line.split(','))

    info = info[1:] #delete first blank item in info

    ##################################LIST CONTAINS DUPLICATES FOR NOW WE WILL JUST DELETE THEM#####################################

    newInfo = []
    addList = []

    for i in info:
        if i[0] not in addList:
            addList.append(i[0])
            newInfo.append(i)




    return newInfo

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
def gkern3D(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen,kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    guass = fi.gaussian_filter(inp, nsig)/kernlen
    max1 = np.amax(guass)
    gauss = guass/max1
    return gauss

def genDensity():
    return np.random.rand(96,96)


def plotDensity(density):
    plt.imshow(density)
    plt.show()

def convertSF(density):
    return np.fft.fftn(density)

def is_number(a):
    # will be True also for 'NaN'
    try:
        number = float(a)
        return True
    except ValueError:
        return False

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

def Remove(duplicate): 
    final_list = [] 
    for num in duplicate: 
        if num not in final_list: 
            final_list.append(num) 
    return final_list 

def buildDensity2D(xSize,ySize,sgInfo,atomSize,sig):

    sg = np.random.randint(low=80,high=100)

    positionTable = []

    for atom in range(0,np.random.randint(1,2)):
        positionTable.append(["0",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0])])

    sym = sgInfo[sg][1:][0]
    cen = sgInfo[sg][2:][0]

    positionTable = applyMultiPos(positionTable)
    positionTable = (buildAndApplySymmetryOpperations(positionTable,sym))
    positionTable = (buildAndApplySymmetryOpperations(positionTable,cen))

    density = np.zeros((xSize,ySize))
    positionTable = np.array(positionTable,dtype=np.dtype(float))[:,[1,2]] %1
    positionTable = np.around(positionTable,decimals=6)
    #positionTable = Remove(positionTable)
    positionTable = np.unique(positionTable,axis=0)

    density = np.pad(density, atomSize, 'constant', constant_values=-1)

    for i in range(0,len(positionTable)):
        x = float(positionTable[i][0])
        y = float(positionTable[i][1])

        if x<-1 or x >2 or y<-1 or y>2:
            continue 
          
      # x = x % 1
      #  y= y % 1

        xi = int(np.round(x*xSize))
        yi = int(np.round(y*ySize))

        atom = gkern2D(atomSize,sig)
        print(atom.shape)

        xmin = int((atomSize)+(xi-(atomSize/2)))
        xmax = int((atomSize)+(xi+(atomSize/2)))

        ymin = int((atomSize)+(yi-(atomSize/2)))
        ymax = int((atomSize)+(yi+(atomSize/2)))

        density[xmin:xmax,ymin:ymax] = density[xmin:xmax,ymin:ymax]  + atom

    density = density[atomSize:-atomSize,atomSize:-atomSize]
    density = np.array(density)
    ff = np.fft.fftn(density).real

    c1 = ff[0:20,0:20]
    c2 = ff[0:20,80:]
    c3 = ff[80:,0:20]
    c4 = ff[80:,80:]

    out1 = np.concatenate((c1,c2),axis=1)
    out2 = np.concatenate((c3,c4),axis=1)
    out = np.concatenate((out1,out2),axis=0)

    out = out-  np.amin(out)
    out = out/np.amax(out)

    return density,np.array([sg-80])


def buildDensity3D(xSize,ySize,zSize,sgInfo,atomSize,sig):
    #Select random spacegroup
    sg = np.random.randint(low=0,high=230)

    #Initilise postion table
    positionTable = []

    #for a random amount of atom assign a random position
    for atom in range(0,np.random.randint(1,2)):
        positionTable.append(["0",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0])])

    #build a list of symmetry operators and centering operations
    sym = sgInfo[sg][1:][0]
    cen = sgInfo[sg][2:][0]

    #positionTable = applyMultiPos(positionTable)
    #Apply symmetry operations
    positionTable = (buildAndApplySymmetryOpperations(positionTable,sym))
    #Apply centering operations
    positionTable = (buildAndApplySymmetryOpperations(positionTable,cen))

    #Generate empty density
    density = np.zeros((xSize,ySize,zSize))
    #Calculate modulus of x,y,z corodinates
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

        atom = gkern3D(atomSize,sig)

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
    

    size_ = 15
    len_ = ff.shape[0]-size_

    c8 = ff[0:size_,0:size_,0:size_]
    c4 = ff[0:size_,0:size_,len_:]
    c7 = ff[0:size_,len_:,0:size_]
    c3 = ff[0:size_,len_:,len_:]
    c1 = ff[len_:,len_:,len_:]
    c2 = ff[len_:,0:size_,len_:]
    c5 = ff[len_:,len_:,0:size_]
    c6 = ff[len_:,0:size_,0:size_]

    out1 = np.concatenate((c1,c2),axis=1)
    out2 = np.concatenate((c3,c4),axis=1)
    out_bottom = np.concatenate((out1,out2),axis=0)

    out1 = np.concatenate((c5,c6),axis=1)
    out2 = np.concatenate((c7,c8),axis=1)
    out_top= np.concatenate((out1,out2),axis=0)
    
    out= np.concatenate((out_bottom,out_top),axis=2)

    #out = out-  np.amin(out)
    #out = out/np.amax(out)
    
    return out,np.array([sg])

def buildModelConv(input_shape,num_classes):
    
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(678, (3, 3), input_shape=input_shape,padding="same",data_format= keras.backend.image_data_format()))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format= keras.backend.image_data_format()))
#    
    model.add(keras.layers.Conv2D(678, (3, 3),padding="same",data_format= keras.backend.image_data_format()))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format= keras.backend.image_data_format()))

##    
    model.add(keras.layers.Conv2D(678, (3, 3),data_format= K.image_data_format()))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),data_format= K.image_data_format()))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    #model.add(keras.layers.Dense(600))
    #model.add(keras.layers.Activation('sigmoid'))
    #model.add(keras.layers.Dense(400))
    #model.add(keras.layers.Activation('sigmoid'))
    #model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(230,activation="sigmoid"))
    model.add(keras.layers.Activation('softmax'))
    return model

def buildModelDense(input_shape,num_classes):
    
    model = keras.Sequential()
    model.add(keras.layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(keras.layers.Dense(30000,input_dim=27000,activation='sigmoid'))
    model.add(keras.layers.Dense(3000,activation='sigmoid'))
    model.add(keras.layers.Dense(230,activation='sigmoid'))

    model.add(keras.layers.Activation("softmax"))   
    return model

def writeGRD(fdat,data):

    out = open("out.grd","w")
    out.write("Title: Put your title \n")                                                         
    out.write(" 5.82930  5.82930  5.82930  90.0000  90.0000  90.0000\n")
    out.write("   "+str(fdat[0])+"    "+str(fdat[1])+"   "+str(fdat[2])+" \n")


    for x in range(0,(fdat[0])):
        for y in range(0,(fdat[1])):
            for z in range(0,(fdat[2])):
                out.write('%.6E' % decimal.Decimal(data[x][y][z]) + "   ")
                if z%6 == 0:
                    out.write("\n")
            out.write("\n")




sgInfo = readSGLib()
"""
dd,sg = buildDensity3D(60,60,60,sgInfo,3,1)
writeGRD(dd.shape,dd)

"""
model = buildModelConv((30,30,30),1)
model = NetSlice(model,'keras', None)
#model.loadModel('3d_230_sg_dense',customObject=None)
print(model.summary())
model.compileModel(keras.optimizers.Adam(lr=0.01), 'categorical_crossentropy', ['accuracy'])
#model.generativeDataTrain(buildDensity, BatchSize=200, Epochs=10,Channel_Ordering=(36,36,1,1),Info=sgInfo)
model.generativeDataTrain(buildDensity3D, BatchSize=500, Epochs=100,Channel_Ordering_Feat=(30,30,30),funcParams=[30,30,30,sgInfo,4,1])
model.saveModel("3d_230_sg_dense_2")
#model.generativeDataTesting(buildDensity3D,SampleNumber=1,Channel_Ordering=(30,30,30),funcParams=[30,30,30,sgInfo,4,1])
