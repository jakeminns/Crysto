import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys
import decimal

class CrystoGen:

    def __init__(self,A=None,B=None,C=None,Alpha=None,Beta=None,Gamma=None,SpaceGroup=None,CrystalSystem=None,AtomTable=None,Info=False, ReciprocalGen=[[-15,15],[-15,15],[-15,15]]):
        
        self.a,self.b,self.c,self.alpha,self.beta,self.gamma,self.sg,self.cT=self.randomCell(A,B,C,Alpha,Beta,Gamma,SpaceGroup,CrystalSystem)
        self.atomTable = self.randomAtoms(AtomTable)
        self.sgInfo = self.readSGLib()
        self.asfInfo = self.readASF()

        if Info:
            print("========== Crystal Info ==========")
            print("")
            print("a:",self.a)
            print("b:",self.b)
            print("c:",self.c)
            print("Alpha:",self.alpha)
            print("Beta:",self.beta)
            print("Gamma:",self.gamma)
            print("Space Group No.:", self.sg+1, "Laue Class:",self.sgInfo[self.sg][2][0],"Space Group:",self.sgInfo[self.sg][1][0])   
            print("")
            print("-------- Atom Table --------")
            
            for atom in self.atomTable:
                print(atom)
            print("")
            print("=================================")
        self.sf = self.buildSF3D(ReciprocalGen[0],ReciprocalGen[1],ReciprocalGen[2])


    def randomAtoms(self,AtomTable):

        if AtomTable == None:

            atomTable = []

            for atom in range(0,np.random.randint(1,4)):
                atomTable.append(["10",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),"1.0"])

            return atomTable

        else:
            return AtomTable

    def randomCell(self,A,B,C,Alpha,Beta,Gamma,sG,cT):

        sg = 0

        if cT != None:
            if cT == 0:
                sg= np.random.randint(low=0,high=2)
            if cT == 1:
                sg= np.random.randint(low=2,high=15)
            if cT == 2:
                sg= np.random.randint(low=15,high=74)
            if cT == 3:
                sg= np.random.randint(low=74,high=142)
            if cT == 4:
                sg= np.random.randint(low=142,high=167)
            if cT == 5:
                sg= np.random.randint(low=167,high=194)
            if cT == 6:
                sg= np.random.randint(low=194,high=230)
        
        
        if sG != None:
            sg = sG
        else:
            sg = np.random.randint(low=0,high=230)

        a,b,c,alpha,beta,gamma = 0.0,0.0,0.0,0.0,0.0,0.0

        if sg>=0 and sg<2:
            cT=0
            a = self.randomCellParam()
            b = self.randomCellParam()
            c = self.randomCellParam()
            alpha = self.randomCellAngle()
            beta = self.randomCellAngle()
            gamma = self.randomCellAngle()
        elif sg>=2 and sg<15:
            a = self.randomCellParam()
            b = self.randomCellParam()
            c = self.randomCellParam()
            alpha = 90.0
            gamma = 90.0
            beta = self.randomCellAngle()
            cT=1
        elif sg>=15 and sg<74:
            a = self.randomCellParam()
            b = self.randomCellParam()
            c = self.randomCellParam()
            alpha = 90.0
            beta = 90.0
            gamma = 90.0
            cT=2
        elif sg>=74 and sg<142:
            a = self.randomCellParam()
            b = a
            c = self.randomCellParam()
            alpha = 90.0
            beta = 90.0
            gamma = 90.0
            cT=3
        elif sg>=142 and sg<167:
            a = self.randomCellParam()
            b = a
            c = self.randomCellParam()
            alpha = 90.0
            beta = 90.0
            gamma = 120.0
            cT=4
        elif sg>=167 and sg<194:
            a = self.randomCellParam()
            b = a
            c = self.randomCellParam()
            gamma = 120.0
            alpha = 90.0
            beta = 90.0
            cT=5
        elif sg>=194 and sg<230:
            a = self.randomCellParam()
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

    def randomCellAngle(self):
        return np.random.uniform(90.5,120.0)

    def randomCellParam(self):
        return np.random.uniform(0.5,30.0)

    def buildSF3D(self,hShape,kShape,lShape,randomCell = False):

        sgInfo = self.sgInfo
        asfInfo = self.asfInfo

        if self.randomCell == False:
            a = self.a
            b = self.b
            c = self.c
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            sg = self.sg
            cT = self.cT
            positionTable = self.atomTable 
        else:
            a,b,c,alpha,beta,gamma,sg,cT=self.randomCell(None,None,None,None,None,None,None,None)
            positionTable = self.randomAtoms(None)

        lambda_ = 0.4
        #build a list of symmetry operators and centering operations  
        sym = sgInfo[sg][3:][0]
        cen = sgInfo[sg][4:][0]

        centMulti = [['x','y','z'],['x+1','y+1','z+1'],['x+1','y+1','z'],['x+1','y','z+1'],['x+1','y','z'],['x','y+1','z+1'],['x','y+1','z'],['x','y','z+1']]
        #Apply symmetry operations
        positionTable = self.buildAndApplySymmetryOpperations(positionTable,cen)
        positionTable = (self.buildAndApplySymmetryOpperations(positionTable,sym))

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
        d,theta = self.quadraticBragg(grid[:,[0]],grid[:,[1]],grid[:,[2]],a,b,c,alpha,beta,gamma,cT,lambda_)
        #theta = np.nan_to_num(theta)
        #Apply self.calcSF across axis for all hkl's calculating all contributions of atoms for each hkl
        ff = np.apply_along_axis(self.calcSF,1,positionTable,*[grid,atomType,theta,asfInfo,lambda_])
        #Sum for all atoms
        ff = np.sum(ff,axis=0)

        ff[np.argmax(ff)] = 0.0
        
        LP = self.LPCorrection(theta*2)
        LP = np.nan_to_num(LP)
        ff = np.multiply(LP,ff,casting='same_kind')
        
        ff= np.abs(ff.real)
        ff = ff/np.amax(ff)
        self.writeHKL(grid,ff,d,theta*2.0,LP)
        self.calculatePowderPattern(grid,ff,180.0,0.01,a,b,c,alpha,beta,gamma,cT,LP,0.01,-0.01,0.0001,0.001,0.5,0.001,4.0)

        #Reshape for grid
        ff = ff.reshape(h.shape[0],k.shape[0],l.shape[0],order="C")

        return ff,sg,a,b,c,alpha,beta,gamma

    def buildAndApplySymmetryOpperations(self,positions, symmetryOpp):
        NewPositions=[]
        for sym in range(0,len(symmetryOpp)):
            for pos in range(0,len(positions)):
                tempPos =[]
                tempPos.append(positions[pos][0])
                tempPos.append(self.applyOpperation(positions[pos],symmetryOpp[sym][0]))
                tempPos.append(self.applyOpperation(positions[pos],symmetryOpp[sym][1]))
                tempPos.append(self.applyOpperation(positions[pos],symmetryOpp[sym][2]))
                tempPos.append(positions[pos][4])
                NewPositions.append(tempPos)
        return NewPositions

    def removeBrackets(self,input):
        
        input = str(input).replace("(","")
        input = str(input).replace(")","")
        
        return input

    def applyOpperation(self,position, symmetryOpp):
        a = symmetryOpp.replace("x",str(self.removeBrackets(position[1])))
        a = a.replace("y",str(self.removeBrackets(position[2])))
        a = a.replace("z",str(self.removeBrackets(position[3])))
        return round(eval(a),3)

    def readSGLib(self):
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

    def readASF(self):
        file_ = np.genfromtxt("atf.txt",unpack=True).T
        return file_

    def quadraticBragg(self,h,k,l,a,b,c,alpha,beta,gamma,cT,lambda_):

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

        return d,(180.0/np.pi)*np.divide(lambda_,np.multiply(2.0,d))

    def calcSF(self,atom,grid,atomType,theta,asfInfo,lambda_):
        atomType = int(atom[0])
        pos = atom[1:4]
        atomB =atom[4]
        asf = self.calculateAtomicScatteringFactor(asfInfo,atomType,theta,lambda_)
        asf = asf.reshape(asf.shape[0],1)
        isoB = self.isoDisplacementFactor(theta,atomB,lambda_)
        sf = np.cos(np.multiply(-2.0*np.pi,(np.multiply(grid[:,[0]],pos[0])+np.multiply(grid[:,[1]],pos[1])+np.multiply(grid[:,[2]],pos[2]))))
        sf = np.multiply(asf,sf,casting='same_kind')
        sf = np.multiply(isoB,sf,casting="same_kind")
        return sf

    def calculateAtomicScatteringFactor(self,file_,element,theta,lambda_):
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

    def isoDisplacementFactor(self,theta,atomB,lambda_):
        rad = np.pi/180.0
        return np.exp(np.multiply(-atomB,(np.divide(np.power(np.sin(theta*rad),2.0),np.power(lambda_,2.0)))))
    
    def LPCorrection(self,theta):
        rad = np.pi/180.0
        return np.divide((1+np.cos(2*rad*theta)**2),np.multiply(np.cos(theta*rad),np.sin(theta*rad)**2))

    def convertToPosition(self,positionTable,a,b,c,alpha,beta,gamma):

        #positionTable = np.array(positionTable)
        rad = np.pi/180.0
        omega = a*b*c*(1-(np.cos(alpha*rad)**2)-(np.cos(beta*rad)**2)-(np.cos(gamma*rad)**2)+2*np.cos(alpha*rad)*np.cos(beta*rad)*np.cos(gamma*rad))**0.5
        convertArr = np.array([[a,b*np.cos(gamma*rad),c*np.cos(beta*rad)],[0,b*np.sin(gamma*rad),c*((np.cos(beta*rad)*np.cos(gamma*rad)-np.cos(gamma*rad))/(np.sin(gamma*rad)))],[0.0,0.0,omega/(a*b*np.sin(gamma*rad))]])
        out = []
        for i in positionTable:
            out.append(np.matmul(np.array(i).transpose(),convertArr).tolist())

        return out

    def calculatePowderPattern(self,grid,ff,_2thetalim,incr,a,b,c,alpha,beta,gamma,cT,LP,U,V,W,Ig,Eta0,X,width):
        
        sf = np.append(grid,ff,axis=1)
        # sf = h,k,l,F
        #Create Empty pattern
        pattern = np.zeros((int(_2thetalim/incr),1))
        patternNoShape = np.zeros((int(_2thetalim/incr),1))

        #Convert hkl into d spacing
        #d = convertHKL2D(sf[:,[0]],sf[:,[1]],sf[:,[2]],a,b,c,alpha,beta,gamma,cT).real
        #Calculate 2theta from d spacing
        d,_2theta= self.quadraticBragg(sf[:,[0]],sf[:,[1]],sf[:,[2]],a,b,c,alpha,beta,gamma,cT,0.4)
        _2theta = _2theta.real*2.0 
        _2theta = np.nan_to_num(_2theta)
        test = np.arange(-10,10,0.01)
        print(test)
        out = self.psPseudoVoigt(0,test,self.psH(U,V,W,Ig,test),Eta0)
        plt.plot(test,out)
        plt.show()
        #Convert width from 2theta to positions in array
        widthPat = int(width/incr)
        #APPENDING COMPLEX NUMBERS TO ARRAY MAKES EVERYTHING COMPLEX
        for i in range(0,_2theta.shape[0]): #_2theta.shape[0]
            pos = int(np.round(_2theta[i].real/incr))

            patternBoundaryMin = pos-widthPat
            patternBoundaryMax = pos+widthPat

            _2thetaSelection = np.linspace(_2theta[i][0]-width,_2theta[i][0]+width,patternBoundaryMax-patternBoundaryMin)
            _2thetaSelection = _2thetaSelection.reshape(_2thetaSelection.shape[0],1)
            #print(pattern[patternBoundaryMin:patternBoundaryMax].shape,pos,patternBoundaryMin,patternBoundaryMax,_2theta[i],np.amin(_2thetaSelection),np.amax(_2thetaSelection),pattern[patternBoundaryMin:patternBoundaryMax].shape[0] )
            pattern[patternBoundaryMin:patternBoundaryMax] = pattern[patternBoundaryMin:patternBoundaryMax] + np.multiply((sf[i,[3]][0].real),self.psPseudoVoigt(_2theta[i][0],_2thetaSelection,self.psH(U,V,W,Ig,_2thetaSelection),Eta0))[:pattern[patternBoundaryMin:patternBoundaryMax].shape[0]]
            #print("PAT",_2theta[i],_2thetaSelection,patternBoundaryMax,patternBoundaryMin)

            patternNoShape[pos] = patternNoShape[pos] + (sf[i,[3]][0].real)

        #self.writeHKL(sf,d,_2theta,LP)

        _2thetaAxis = np.arange(0,_2thetalim,incr)
        self.writePowderPlot(_2thetaAxis,pattern)
        plt.plot(_2thetaAxis,pattern,"r")
        #plt.plot(_2thetaAxis,patternNoShape)
        plt.show()

    def psGaussian(self,x,H):
        a = (2.0/H)*(np.log(2.0)/np.pi)**0.5
        b = (4*np.log(2.0))/(H**2.0)
        return np.multiply(a,np.exp(np.multiply(-b,np.power(x,2.0))))

    def psLorentz(self,x,H):
        a = 2/(np.pi*H)
        b = 4.0/(np.power(H,2.0))
        return np.divide(a,(1+np.multiply(b,np.power(x,2.0))))

    def psEta(self,Eta0,X,_2theta):
        return Eta0+np.mulitply(X,_2theta)

    def psH(self,U,V,W,Ig,_2theta):
        theta = np.divide(_2theta,2.0)
        rad = np.pi/180.0
        return np.power(np.multiply(U,np.power(np.tan(theta*rad),2.0))+np.multiply(V,np.tan(rad*theta))+W+np.divide(Ig,(np.power(np.cos(theta*rad),2.0))),0.5)

    def psPseudoVoigt(self,_2theta0,_2theta,H,Eta):
        theta = _2theta-_2theta0
        return np.multiply(Eta,self.psLorentz(theta,H))+np.multiply((1-Eta),self.psGaussian(theta,H))
    def SFtoStruct(self,sf):
        sf = np.fft.fftshift(sf)
        sf = np.fft.fftn(sf).real1
        return (sf)

    def writePowderPlot(self,axis,intensity):
        file_ = open("plot.dat","w")

        for theta, int_ in zip(axis,intensity):
            file_.write(str('{:06f}'.format(theta))+"   "+str('{:06f}'.format(int_[0]))+"\n")

    def writeGRD(self,fdat,data,name,a,b,c,alpha,beta,gamma,Setting=0):

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

    def writeHKL(self,grid,ff,d,theta,LP):

        out = open("out.hkl","w")
        out.write("#  h    k    l     |F|        d       2theta       LP   \n")
        for i in range(0,grid.shape[0]):
            out.write("  "+str('{:3d}'.format(int(grid[i][0])))+"  "+str('{:3d}'.format(int(grid[i][1])))+"  "+str('{:3d}'.format(int(grid[i][2])))+"  "+str('{:06f}'.format(ff[i][0].real))+"  "+str('{:06f}'.format(d[i][0]))+"  "+str('{:06f}'.format(theta[i][0]))+"  "+str('{:06f}'.format(LP[i][0]))+"\n")

    def genrateTrainingData(num,funcParams):

        dataFeat = []
        dataLabel = []

        for i in range(0,num):
            if i % 100 == 0:
                print(i)
            item = buildSF3D(*funcParams,True)
            dataFeat.append(item[0])
            dataLabel.append(np.array([item[1],item[2],item[3],item[4],item[5],item[6],item[7]]))

        dataFeat = np.array(dataFeat)
        dataLabel = np.array(dataLabel)

        np.save("feat.npy",dataFeat)
        np.save("label.npy",dataLabel)




cell = CrystoGen(CrystalSystem=6,SpaceGroup=200,Info=True)

#dd,s,a,b,c,alpha,beta,gamma = buildSF3D([-10,10],[-10,10],[-10,10],sgInfo,asfInfo)
#self.writeGRD(dd.shape,dd,"density",a,b,c,alpha,beta,gamma,Setting=0)
#self.writeGRD(ff.shape,ff,"struct",a,b,c,alpha,beta,gamma,Setting=1)

