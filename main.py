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
    a,b,c,alpha,beta,gamma,sg = randomCell(cT=5,A=8.88, B=11.99,C=5.3,Alpha= 90.0,Beta= 90.0,Gamma= 90.0,sG= 72)


    #Initilise postion table
    positionTable = []
    #positionTable.append(["0",str("0"),str(0),str(0)])

    #for a random amount of atom assign a random position
    for atom in range(0,np.random.randint(1,2)):
        positionTable.append(["0",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0])])
    positionTable = [["0","0","0","0"],["0","0.123","0.321","0.213"]]
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
    print(positionTable,len(positionTable))
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
    for sym in range(0,len(symmetryOpp)):
        for pos in range(0,len(positions)):
            tempPos =[]
            tempPos.append(positions[pos][0])
            tempPos.append(applyOpperation(positions[pos],symmetryOpp[sym][0]))
            tempPos.append(applyOpperation(positions[pos],symmetryOpp[sym][1]))
            tempPos.append(applyOpperation(positions[pos],symmetryOpp[sym][2]))
            tempPos.append(positions[pos][4])
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

def writeGRD(fdat,data,name,a,b,c,alpha,beta,gamma,Setting=0):

    out = open(name+".grd","w")
    out.write("Title: Put your title \n")
    if Setting==0:
        out.write(" "+str(fdat[0])+"  "+str(fdat[1])+"  "+str(fdat[2])+"  "+str(90)+"  "+str(90)+"  "+str(90)+"\n")
    else:
        out.write(" "+str(a)+"  "+str(b)+"  "+str(c)+"  "+str(alpha)+"  "+str(beta)+"  "+str(gamma)+"\n")
    
    cell = np.array(fdat)/10.0                              

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

        for a in ["","+1"]:
            for b in ["","+1"]:   
                for c in ["","+1"]:  
                    item=[]
                    print("x",a,"y",b,"z",c)
                    item.append("0")
                    item.append(calculateSymOp(str(i[1]),"x","x"+a))  
                    item.append(calculateSymOp(str(i[2]),"y","y"+b))       
                    item.append(calculateSymOp(str(i[3]),"z","z"+c))  
                    newPos.append((item) )  

    return (newPos)

def randomCellAngle():
    return np.random.uniform(90.5,120.0)

def randomCellParam():
    return np.random.uniform(0.5,30.0)

def randomCell(cT = None,sG=None,A=None,B=None,C=None,Alpha=None,Beta=None,Gamma=None):

    if cT == None:
        cT = np.random.randint(low=0,high=7) # Choose Cell Type 0=Tri,1=Mono,2=Orth,3=Tetr,4=Triagonal,5=Hex,6=Cubic
    
    a,b,c,alpha,beta,gamma = 0.0,0.0,0.0,0.0,0.0,0.0
    sg = 0

    if sG != None:
        sg = sG
    else:
        sg = np.random.randint(low=0,high=230)

    if sg>=0 and sg<2:
        cT=0
        a = randomCellParam()
        b = randomCellParam()
        c = randomCellParam()
        alpha = randomCellAngle()
        beta = randomCellAngle()
        gamma = randomCellAngle()
    elif sg>=2 and sg<15:
        a = randomCellParam()
        b = randomCellParam()
        c = randomCellParam()
        alpha = 90.0
        gamma = 90.0
        beta = randomCellAngle()
        cT=1
    
    elif sg>=15 and sg<74:
        a = randomCellParam()
        b = randomCellParam()
        c = randomCellParam()
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
        cT=2

    elif sg>=74 and sg<142:
        a = randomCellParam()
        b = a
        c = randomCellParam()
        alpha = 90.0
        beta = 90.0
        gamma = 90.0
        cT=3

    elif sg>=142 and sg<167:
        a = randomCellParam()
        b = a
        c = randomCellParam()
        alpha = 90.0
        beta = 90.0
        gamma = 120.0
        cT=4

    elif sg>=167 and sg<194:
        a = randomCellParam()
        b = a
        c = randomCellParam()
        gamma = 120.0
        alpha = 90.0
        beta = 90.0
        cT=5

    elif sg>=194 and sg<230:
        a = randomCellParam()
        b = a
        c = a
        gamma = 90.0
        alpha = 90.0
        beta = 90.0
        cT=6

    else:       
        print("Incorrect Cell Type Chosen")
        sys.exit()   


    
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

    return a,b,c,alpha,beta,gamma,sg,cT

def convertToPosition(positionTable,a,b,c,alpha,beta,gamma):

    #positionTable = np.array(positionTable)
    rad = np.pi/180.0
    omega = a*b*c*(1-(np.cos(alpha*rad)**2)-(np.cos(beta*rad)**2)-(np.cos(gamma*rad)**2)+2*np.cos(alpha*rad)*np.cos(beta*rad)*np.cos(gamma*rad))**0.5
    convertArr = np.array([[a,b*np.cos(gamma*rad),c*np.cos(beta*rad)],[0,b*np.sin(gamma*rad),c*((np.cos(beta*rad)*np.cos(gamma*rad)-np.cos(gamma*rad))/(np.sin(gamma*rad)))],[0.0,0.0,omega/(a*b*np.sin(gamma*rad))]])
    out = []
    for i in positionTable:
        out.append(np.matmul(np.array(i).transpose(),convertArr).tolist())

    return out
def quadraticBragg(h,k,l,a,b,c,alpha,beta,gamma,cT,lambda_):
    print("QUAD",cT)
    print(h[0],k[0],l[0])
    rad = np.pi/180.0

    if cT == 0:
        V = a*b*c*(1+2*np.cos(rad*alpha)*np.cos(rad*beta)*np.cos(rad*gamma)-(np.cos(alpha*rad)**2.0)-(np.cos(beta*rad)**2.0)-(np.cos(gamma*rad)**2.0))**0.5

        aS = (1/V)*b*c*np.sin(rad*alpha)
        cosAS = (np.cos(beta*rad)*np.cos(gamma*rad)-np.cos(alpha*rad))/(np.sin(rad*beta)*np.sin(rad*gamma))

        bS = (1/V)*a*c*np.sin(rad*beta)
        cosBS = (np.cos(gamma*rad)*np.cos(alpha*rad)-np.cos(beta*rad))/(np.sin(rad*gamma)*np.sin(rad*alpha))

        cS = (1/V)*a*b*np.sin(rad*gamma)
        cosGS = (np.cos(alpha*rad)*np.cos(beta*rad)-np.cos(gamma*rad))/(np.sin(rad*alpha)*np.sin(rad*beta))

        const = (lambda_**2.0)/(4.0)
        print(V,aS,bS,cS,cosAS,cosBS,cosGS,const)
        hkl = np.multiply(np.power(h,2.0),aS**2.0)+np.multiply(np.power(k,2.0),bS**2.0)+np.multiply(np.power(l,2.0),cS**2.0)+(2.0*np.multiply(k,l)*bS*cS*cosAS)+(2.0*np.multiply(l,h)*cS*aS*cosBS)+(2.0*np.multiply(l,h)*cS*aS*cosBS)+(2.0*np.multiply(h,k)*aS*bS*cosGS)
        #_2t = np.multiply(const,hkl)
        #_2t = np.arcsin(np.power(_2t,0.5))

    elif cT==1:
        const = (lambda_**2.0)/(4.0)
        h_ = np.divide(np.power(h,2.0),np.multiply((a**2.0),np.power(np.sin(beta*rad),2.0)))
        k_ = np.divide(np.power(k,2.0),b**2.0)
        l_ = np.divide(np.power(l,2.0),np.multiply((c**2.0),np.power(np.sin(beta*rad),2.0)))
        a_ = np.divide((2.0*np.multiply(h,l))*np.cos(rad*beta),(a*c*(np.power(np.sin(beta*rad),2.0))))
        hkl = h_+k_+l_-a_
        #_2t = np.multiply(const,hkl)
        
        #_2t = np.arcsin(np.power(_2t,0.5))
    elif cT==2:
        const = (lambda_**2.0)/(4.0)
        hkl = np.divide(np.power(h,2.0),a**2.0)+np.divide(np.power(k,2.0),b**2.0)+np.divide(np.power(l,2.0),c**2.0)
        #_2t = np.multiply(const,hkl)
        #_2t = np.arcsin(np.power(_2t,0.5))
    elif cT==3:

        const = (lambda_**2.0)/(4.0)
        const2 = 1.0/(a**2.0)
        hkl = (np.power(h,2.0)+np.power(k,2.0)+(np.multiply((a/c)**2.0,np.power(l,2.0))))
        hkl = np.multiply(const2,hkl)
        #_2t = np.multiply(const,hkl)
        #_2t = np.arcsin(np.power(_2t,0.5))

    elif cT==4 or cT==5:

        const = (lambda_**2.0)/(4.0)
        const2 = 1.0/(a**2.0)
        hkl = np.multiply((4.0/3.0),(np.power(h,2.0)+np.power(k,2.0)+np.multiply(h,k)))+(np.multiply((a/c)**2.0,np.power(l,2.0)))
        hkl = np.multiply(const2,hkl)
        #_2t = np.multiply(const,hkl)
        #_2t = np.arcsin(np.power(_2t,0.5))

    elif cT==9:
        const = (lambda_**2.0)/(4.0)

        hkl = (np.power(h,2.0)+np.power(k,2.0)+np.power(l,2.0))
        hkl2 = np.multiply(h,k)+np.multiply(k,l)+np.multiply(h,l)
        hklTop = np.multiply(hkl,np.sin(rad*gamma)**2.0)+np.multiply((2.0*hkl2),(np.cos(gamma*rad)**2.0))-np.cos(gamma*rad)
        hklBot = (a**2.0)*(1-(3*np.cos(rad*gamma)**2.0)+(2.0*np.cos(gamma*rad)**3.0))
        hkl = np.divide(hklTop,hklBot)
        #_2t = np.multiply(const,hkl)
        #_2t = np.arcsin(np.power(_2t,0.5))


    elif cT==6:

        const = (lambda_**2.0)/(4.0)
        const2 = 1.0/(a**2.0)
        hkl = (np.power(h,2.0)+np.power(k,2.0)+np.power(l,2.0))
        hkl = np.multiply(const2,hkl)

        #_2t = np.multiply(const,hkl)
        #_2t = np.arcsin(np.power(_2t,0.5))

    else:
        print("Incorrect Cell Type Chosen")
        sys.exit()  
    d = np.power(np.divide(1.0,hkl),0.5)

    for i in range(0,100):
        print(h[i],k[i],l[i], d[i],(180.0/np.pi)*np.divide(lambda_,np.multiply(2.0,d[i])))
#np.power(np.divide(1.0,hkl),0.5)
    return d,(180.0/np.pi)*np.divide(lambda_,np.multiply(2.0,d))

def convertHKL2D(h,k,l,a,b,c,alpha,beta,gamma,cT):

    rad = np.pi/180.0

    if cT == 0:
        print("No HKL to D caluclated yet")
        sys.exit()  
    elif cT==1:
        d = (1.0/np.sin(beta*rad)**2.0)*(((h**2.0)/(a**2.0))+(((k**2.0)*np.sin(rad*beta)**2.0)/(b**2.0))+((l**2.0)/(c**2.0))-((2.0*h*l*np.cos(beta*rad))/a*c))
        print("d",d)

        d= (1.0/d)**0.5
    elif cT==2:
        d = ((h**2.0)/(a**2.0))+((k**2.0)/(b**2.0))+((l**2.0)/(c**2.0))
        print("d",d)
        d= (1.0/d)**0.5
    elif cT==3:
        d = (((h**2.0)+(k**2.0))/a**2.0)+((l**2.0)/(c**2.0))
        print("d",d)

        d=(1.0/d)**0.5
    elif cT==4:
        print("No HKL to D caluclated yet")
        sys.exit()  
    elif cT==5:
        d = (4.0/3.0)*(((h**2.0)+h*k+(k**2.0))/(a**2.0))+((l**2.0)/(c**2.0))
        d= (1/d)**0.5
    elif cT==6:
        d = ((h**2)+(k**2)+(l**2))/(a**2)
        d= (1/d)**0.5
    else:
        print("Incorrect Cell Type Chosen")
        sys.exit()   

    print("dffff",d)

    return d
def d2theta(d,lambd):
    #check for non numbers
    d = np.nan_to_num(d)
    d = (180.0/np.pi)*np.arcsin(lambd/(2.0*d))
    print("thetaaaa",d)
    return d

def buildSF3D(hShape,kShape,lShape,sgInfo,asfInfo):

    lambda_ = 0.4
   #a,b,c,alpha,beta,gamma,sg,cT = randomCell(A=1.5814,B= 16.9829,C= 9.166782,Alpha= 90.0,Beta= 90.0,Gamma= 90.0,sG= 28)
    #Generaate lattice parameters and space group
    a,b,c,alpha,beta,gamma,sg,cT = randomCell(sG=1)
    #Initilise postion table
    positionTable = []
    #for a random amount of atom assign a random position
    for atom in range(0,np.random.randint(1,4)):
        positionTable.append(["10",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),"1.0"])
    #positionTable = [["10","0.0","0.0","0.0"],["10","0.123","0.321","0.231"]]
    #positionTable = [["10","0.0","0.0","0.0"]]
    positionTable = [['10', '0.17421616359686776', '0.009086319277717747', '0.616727098909161',"1.0"]]
    print("pos",positionTable)
    #build a list of symmetry operators and centering operations  
    sym = sgInfo[sg][3:][0]
    cen = sgInfo[sg][4:][0]
    laue = sgInfo[sg][2][0]
    name = sgInfo[sg][1][0]
    print("params",a,b,c,alpha,beta,gamma,sg,cT,name)

    centMulti = [['x','y','z'],['x+1','y+1','z+1'],['x+1','y+1','z'],['x+1','y','z+1'],['x+1','y','z'],['x','y+1','z+1'],['x','y+1','z'],['x','y','z+1']]
    #Apply symmetry operations
    positionTable = buildAndApplySymmetryOpperations(positionTable,cen)

    positionTable = (buildAndApplySymmetryOpperations(positionTable,sym))

    #Apply centering operations
    positionTable = np.array(positionTable,dtype=np.dtype(float))
    atomType = positionTable[:,[0]]
    atomISO = positionTable[:,[4]]
    positionTable = positionTable[:,[1,2,3]]%1

    positionTable = np.append(atomType,positionTable,axis=1)
    positionTable = np.append(positionTable,atomISO,axis=1)
    #Find absolute
    positionTable = np.abs(positionTable)
    #Round results for removal of duplicates
    positionTable = np.around(positionTable,decimals=6) 
    #Remove duplicates
    positionTable = np.unique(positionTable,axis=0)

    h = np.arange(hShape[0],hShape[1])
    k = np.arange(kShape[0],kShape[1])
    l = np.arange(lShape[0],lShape[1])
    #Generate all combinations of hkl
    grid = np.array(np.meshgrid(h,k,l)).T.reshape(h.shape[0]*k.shape[0]*l.shape[0],3,order="C")
    #Generate empty struture factor density
    ff = np.full((len(positionTable),h.shape[0]*k.shape[0]*l.shape[0],1), 0.0+0.0j)
    print(grid)
    d,theta = quadraticBragg(grid[:,[0]],grid[:,[1]],grid[:,[2]],a,b,c,alpha,beta,gamma,cT,lambda_)
    print("here",theta)
    #theta = np.nan_to_num(theta)
    #Apply calcSF across axis for all hkl's calculating all contributions of atoms for each hkl
    ff = np.apply_along_axis(calcSF,1,positionTable,*[grid,atomType,theta,asfInfo,lambda_])
    #Sum for all atoms
    ff = np.sum(ff,axis=0)

    ff[np.argmax(ff)] = 0.0
    
    LP = LPCorrection(theta*2)
    LP = np.nan_to_num(LP)
    ff = np.multiply(LP,ff,casting='same_kind')
    
    ff= np.abs(ff.real)
    ff = ff/np.amax(ff)
    writeHKL(grid,ff,d,theta*2.0,LP)
    calculatePowderPattern(grid,ff,180.0,0.01,a,b,c,alpha,beta,gamma,cT,LP)

    #Reshape for grid
    ff = ff.reshape(h.shape[0],k.shape[0],l.shape[0],order="C")

    return ff,sg,a,b,c,alpha,beta,gamma



def calculatePowderPattern(grid,ff,_2thetalim,incr,a,b,c,alpha,beta,gamma,cT,LP):
    
    sf = np.append(grid,ff,axis=1)
    # sf = h,k,l,F
    #remove f(000) reflection
    #sf = np.delete(sf,np.argmax(sf[:,[3]]),0)
    multi = np.zeros((sf.shape[0],1))
    #Create Empty pattern
    pattern = np.zeros((int(_2thetalim/incr),1))
    #Convert hkl into d spacing
 #   d = convertHKL2D(sf[:,[0]],sf[:,[1]],sf[:,[2]],a,b,c,alpha,beta,gamma,cT).real
    #Calculate 2theta from d spacing
    d,_2theta= quadraticBragg(sf[:,[0]],sf[:,[1]],sf[:,[2]],a,b,c,alpha,beta,gamma,cT,0.4)
    _2theta = _2theta.real*2.0 
    _2theta = np.nan_to_num(_2theta)
    #APPENDING COMPLEX NUMBERS TO ARRAY MAKES EVERYTHING COMPLEX
    for i in range(0,_2theta.shape[0]):
        pos = int(np.round(_2theta[i].real/incr))
        pattern[pos] = pattern[pos] + (sf[i,[3]][0].real)

    #writeHKL(sf,d,_2theta,LP)

    _2thetaAxis = np.arange(0,_2thetalim,incr)
    writePowderPlot(_2thetaAxis,pattern)
    plt.plot(_2thetaAxis,pattern)
    plt.show()

def SFtoStruct(sf):
    sf = np.fft.fftshift(sf)
    sf = np.fft.fftn(sf).real1
    return (sf)
def writePowderPlot(axis,intensity):
    file_ = open("plot.dat","w")

    for theta, int_ in zip(axis,intensity):
        file_.write(str('{:06f}'.format(theta))+"   "+str('{:06f}'.format(int_[0]))+"\n")
def generateAtomicScateringFactors():
    file_ = np.genfromtxt("atf.txt",unpack=True).T
    return file_

def calculateAtomicScatteringFactor(file_,element,theta,lambda_):
    params = file_[element][1:9]
    rad = np.pi/180.0

    params = params.reshape(4,2)
    c = file_[element][9]
    theta1 = np.sin(theta*rad)/lambda_

    expo = np.power(theta1,2.0)
    expo = -np.outer(params[:,[1]],expo).T
    f = np.exp(expo).dot(params[:,[0]])
    f = f + c
    return f

def LPCorrection(theta):
    rad = np.pi/180.0
    return np.divide((1+np.cos(2*rad*theta)**2),np.multiply(np.cos(theta*rad),np.sin(theta*rad)**2))

def calcSF(atom,grid,atomType,theta,asfInfo,lambda_):
    #VFunc = np.vectorize(SF)
    atomType = int(atom[0])
    pos = atom[1:4]
    atomB =atom[4]
    print("ATOM",atomType,pos,atomB)
    asf = calculateAtomicScatteringFactor(asfInfo,atomType,theta,lambda_)
    asf = asf.reshape(asf.shape[0],1)
    isoB = isoDisplacementFactor(theta,atomB,lambda_)
    t1 = np.cos(np.multiply(-2.0*np.pi,(np.multiply(grid[:,[0]],pos[0])+np.multiply(grid[:,[1]],pos[1])+np.multiply(grid[:,[2]],pos[2]))))
    print("iSo",isoB)
    t1 = np.multiply(asf,t1,casting='same_kind')
    t1 = np.multiply(isoB,t1,casting="same_kind")
    return t1


def isoDisplacementFactor(theta,atomB,lambda_):
    rad = np.pi/180.0
    return np.exp(np.multiply(-atomB,(np.divide(np.power(np.sin(theta*rad),2.0),np.power(lambda_,2.0)))))

def genrateTrainingData(num,funcParams):

    dataFeat = []
    dataLabel = []

    for i in range(0,num):
        if i % 100 == 0:
            print(i)
        item = buildSF3D(*funcParams)
        dataFeat.append(item[0])
        dataLabel.append(np.array([item[1],item[2],item[3],item[4],item[5],item[6],item[7]]))

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
    model.add(keras.layers.DensapplyMultiPospee(59,activation='sigmoid'))

    return model


def writeHKL(grid,ff,d,theta,LP):

    out = open("out.hkl","w")
    out.write("#  h    k    l     |F|        d       2theta       LP   \n")
    for i in range(0,grid.shape[0]):
        out.write("  "+str('{:3d}'.format(int(grid[i][0])))+"  "+str('{:3d}'.format(int(grid[i][1])))+"  "+str('{:3d}'.format(int(grid[i][2])))+"  "+str('{:06f}'.format(ff[i][0].real))+"  "+str('{:06f}'.format(d[i][0]))+"  "+str('{:06f}'.format(theta[i][0]))+"  "+str('{:06f}'.format(LP[i][0]))+"\n")


sgInfo = readSGLib()
asfInfo = generateAtomicScateringFactors()


dd,s,a,b,c,alpha,beta,gamma = buildSF3D([-10,10],[-10,10],[-10,10],sgInfo,asfInfo)
writeGRD(dd.shape,dd,"density",a,b,c,alpha,beta,gamma,Setting=0)
#writeGRD(ff.shape,ff,"struct",a,b,c,alpha,beta,gamma,Setting=1)

"""
#writeHKL(dd,[0,15],[-15,15],[-15,15])
#
#writeGRD(ff.shape,ff,"reciprocal_cubic",a,b,c,alpha,beta,gamma)


genrateTrainingData(10000,[[0,15],[-15,15],[-15,15],sgInfo,asfInfo])


feat = np.load("feat.npy")
label = np.load("label.npy")
#print(feat.shape,label.shape)

model = buildModelConv((15,30,30, 1),1)
data = DataSlice(Features=feat,Labels=label,Channel_Features=[15,30,30,1],Shuffle = False,Split_Ratio=0.85)
data.channelOrderingFormatFeatures(15,30,30,1)
data.oneHot(36)
model = NetSlice(model,'keras', data)
#model.loadModel('3d_Cubic_conv_simple',customObject=None)
#print(model.summary())
model.compileModel(tf.train.AdamOptimizer(), 'categorical_crossentropy', ['accuracy'])
#print(model.summary())
model.trainModel(Epochs=20,Batch_size=1000,Verbose=1)
#model.generativeDataTrain(buildDensity, BatchSize=200, Epochs=10,Channel_Ordering=(36,36,1,1),Info=sgInfo)
#model.generativeDataTrain(buildDensity3D, BatchSize=3000, Epochs=10,Channel_Ordering_Feat=(30,30,30),funcParams=[60,60,60,sgInfo,4,1],OneHot=36)
model.saveModel("3d_Cubic_conv_simple")
"""