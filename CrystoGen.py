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
        self.pattern = ""
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

        self.sf,self.sfComplex,self.grid,self.d,self.theta,sg,a,b,c,alpha,beta,gamma = self.buildSF3D(ReciprocalGen[0],ReciprocalGen[1],ReciprocalGen[2])

        self.pattern = self.calculatePowderPattern(180.0,0.01,0.1,-0.0001,0.0001,0.001,0.5,0.001,8.0,1.0)

       # self.writeHKL()

    def randomAtoms(self,AtomTable):

        if AtomTable == None:

            atomTable = []

            for atom in range(0,np.random.randint(1,4)):
                atomTable.append(["10",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),"1.0","1.0"])

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

    def buildSF3D(self,hShape,kShape,lShape,randomCell = False,FormFactor=True,TempFactor=True,LorentzP=True,Normalize=True,RemoveF000=True):

        sgInfo = self.sgInfo
        asfInfo = self.asfInfo

        if randomCell == False:
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
        #positionTable = self.buildAndApplySymmetryOpperations(positionTable,centMulti)

        positionTable = self.buildAndApplySymmetryOpperations(positionTable,cen)
        positionTable = (self.buildAndApplySymmetryOpperations(positionTable,sym))
        #Apply centering operations
        positionTable = np.array(positionTable,dtype=np.dtype(float))
        atomType = positionTable[:,[0]]
        atomOcc = positionTable[:,[4]]
        atomISO = positionTable[:,[5]]
        positionTable = positionTable[:,[1,2,3]]%1



        positionTable = np.append(atomType,positionTable,axis=1)
        positionTable = np.append(positionTable,atomOcc,axis=1)

        positionTable = np.append(positionTable,atomISO,axis=1)
        #Find absolute
        positionTable = np.abs(positionTable)
        #Round results for removal of duplicates
        positionTable = np.around(positionTable,decimals=6) 
        #Remove duplicates
        positionTable = np.unique(positionTable,axis=0)
        #print(a,b,c,alpha,beta,gamma,sg,positionTable,len(positionTable))
        h = np.arange(hShape[0],hShape[1])
        k = np.arange(kShape[0],kShape[1])
        l = np.arange(lShape[0],lShape[1])
        #Generate all combinations of hkl
        grid = np.array(np.meshgrid(h,k,l)).T.reshape(h.shape[0]*k.shape[0]*l.shape[0],3,order="C")
        #Generate empty struture factor density
        ffComplex = np.full((len(positionTable),h.shape[0]*k.shape[0]*l.shape[0],1), 0.0+0.0j)
        d,theta = self.quadraticBragg(grid[:,[0]],grid[:,[1]],grid[:,[2]],lambda_)
        #multi = self.calculateMultiplicity()

        #Apply self.calcSF across axis for all hkl's calculating all contributions of atoms for each hkl
        ffComplex = np.apply_along_axis(self.calcSF,1,positionTable,*[grid,atomType,theta,asfInfo,lambda_,FormFactor,TempFactor])
        #Sum for all atoms
        ffComplex = np.sum(ffComplex,axis=0)

        if RemoveF000 == True:
            ffComplex[np.argmax(ffComplex)] = 0.0
             
        ffAbs= np.absolute(ffComplex)
        ffAbs = np.power(ffAbs,2.0)

        #if LorentzP == True:
            #ffAbs = np.multiply(LP,ffAbs,casting='same_kind')
        
        if Normalize == True:
            ffAbs = ffAbs/np.amax(ffAbs)
            #ffComplex = ffComplex/np.amax(ffComplex)

        #Reshape for grid
        ffComplex = ffComplex.reshape(h.shape[0],k.shape[0],l.shape[0],order="C")
        ffAbs = ffAbs.reshape(h.shape[0],k.shape[0],l.shape[0],order="C")

        return ffAbs,ffComplex,grid,d,theta,sg,a,b,c,alpha,beta,gamma

    def buildAndApplySymmetryOpperations(self,positions, symmetryOpp):
        NewPositions=[]
        for sym in range(0,len(symmetryOpp)):
            for pos in range(0,len(positions)):
                tempPos =[]
                tempPos.append(positions[pos][0]) #Atom Type
                tempPos.append(self.applyOpperation(positions[pos],symmetryOpp[sym][0])) #X
                tempPos.append(self.applyOpperation(positions[pos],symmetryOpp[sym][1])) #Y
                tempPos.append(self.applyOpperation(positions[pos],symmetryOpp[sym][2])) #Z
                tempPos.append(positions[pos][4]) #Occ
                tempPos.append(positions[pos][5]) #Iso
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

    def quadraticBragg(self,h,k,l,lambda_):

        a = self.a
        b = self.b
        c = self.c
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        cT = self.cT

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

    def calcSF(self,atom,grid,atomType,theta,asfInfo,lambda_,FormFactor,TempFactor):
        atomType = int(atom[0])        #sf = np.cos(np.multiply(-2.0*np.pi,(np.multiply(grid[:,[0]],pos[0])+np.multiply(grid[:,[1]],pos[1])+np.multiply(grid[:,[2]],pos[2]))))

        pos = atom[1:4]
        atomB =atom[5]
        atomOcc = atom[4]
        #sf = np.cos(np.multiply(-2.0*np.pi,(np.multiply(grid[:,[0]],pos[0])+np.multiply(grid[:,[1]],pos[1])+np.multiply(grid[:,[2]],pos[2]))))
        sf = np.exp(np.multiply(-2.0j*np.pi,(np.multiply(grid[:,[0]],pos[0])+np.multiply(grid[:,[1]],pos[1])+np.multiply(grid[:,[2]],pos[2]))))

        if FormFactor == True:
            asf = self.calculateAtomicScatteringFactor(asfInfo,atomType,theta,lambda_)
            asf = asf.reshape(asf.shape[0],1)

            sf = np.multiply(asf,sf,casting='same_kind')

        if TempFactor == True:
            isoB = self.isoDisplacementFactor(theta,atomB,lambda_)

            sf = np.multiply(isoB,sf,casting="same_kind")

        #sf = np.multiply(sf,atomOcc)
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
        #LP = np.divide(1.0+np.power(np.cos(theta*rad),2.0),2.0*np.sin(theta*rad))
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

    def generateHKLGrid(self,boundaries):
        hShape = boundaries[0]
        kShape = boundaries[1]
        lShape = boundaries[2]

        h = np.arange(hShape[0],hShape[1])
        k = np.arange(kShape[0],kShape[1])
        l = np.arange(lShape[0],lShape[1])
        #Generate all combinations of hkl
        return np.array(np.meshgrid(h,k,l)).T.reshape(h.shape[0]*k.shape[0]*l.shape[0],3,order="C")

    def calculatePowderPattern(self,_2thetalim,incr,U,V,W,Ig,Eta0,X,width,ScaleFactor):
        
        sf = np.append(self.grid,self.sf.reshape(self.sf.shape[0]*self.sf.shape[1]*self.sf.shape[2],1),axis=1)
        # sf = h,k,l,F
        #Empty Pattern arrays 
        pattern = np.ones((int(_2thetalim/incr),1))
        patternNoShape = np.ones((int(_2thetalim/incr),1))

        d = self.d
        _2theta= self.theta
        _2theta = _2theta.real*2.0
        sf = np.append(sf,_2theta,axis=1)

        #Convert width from 2theta to positions in array
        widthPat = int(width/incr)
        count = 0
        for i in range(0,_2theta.shape[0]): #_2theta.shape[0]

            #Locate position (item number) of 2theta in pattern array
            pos = int(np.round(_2theta[i][0]/incr))
     
            if pos < pattern.shape[0]:
                _2thetaZero = _2theta[i][0]
                #max min pattern values for all 2theta values between width
                patternBoundaryMin = pos-widthPat
                patternBoundaryMax = pos+widthPat
            
                _2thetaMin = _2thetaZero-width
                _2thetaMax = _2thetaZero+width

                _2thetaSelection = np.linspace(_2thetaMin,_2thetaMax,patternBoundaryMax-patternBoundaryMin)
         
                _2thetaSelection = _2thetaSelection.reshape(_2thetaSelection.shape[0],1)
                H = self.psH(U,V,W,Ig,_2thetaZero)
                
                ps =  self.psPseudoVoigt(_2thetaZero,_2thetaSelection,H,Eta0)
                newInt = np.multiply((sf[i,[3]][0].real),ps)
                patternNoShape[pos] = patternNoShape[pos] + sf[i,[3]][0].real
                if patternBoundaryMin <0:
                    newInt = newInt[np.abs(patternBoundaryMin):]
                    patternBoundaryMin = 0

                if patternBoundaryMax > pattern.shape[0]:
                    patternBoundaryMax = pattern.shape[0]
                    newInt = newInt[0:int(patternBoundaryMax-patternBoundaryMin)]

                if pattern[patternBoundaryMin:patternBoundaryMax].shape[0] != 0 or pattern.shape[0]>patternBoundaryMin:
                    pattern[patternBoundaryMin:patternBoundaryMax] = np.add(pattern[patternBoundaryMin:patternBoundaryMax], newInt)

        pattern = pattern/np.amax(pattern)
        pattern = np.multiply(pattern,ScaleFactor)

        _2thetaAxis = np.arange(0,_2thetalim,incr)

        _2thetaAxisRe = _2thetaAxis.reshape(_2thetaAxis.shape[0],1)[30:]
        LP_ = self.LPCorrection(_2thetaAxisRe/2.0)

        #LP_[np.argmax(LP_)] = 0.0
        #LP_[:500] = 1.0
        pattern = np.multiply(LP_,pattern[30:])
        #self.writePowderPlot(_2thetaAxis[30:3000],pattern[30:3000])
        #plt.plot(_2thetaAxis[:3000],pattern[:3000],"r")
        #plt.plot(_2thetaAxis[:3000],patternNoShape[:3000])
        #plt.xlabel("2theta")
        #plt.ylabel("Intensity (Normalised)")
        #plt.show()
        return pattern[:3000]

        
    def psGaussian(self,x,H):
        a = (1.0/H)*(4.0*np.log(2.0)/np.pi)**0.5
        b = (4*np.log(2.0))/(H**2.0)
        return np.multiply(a,np.exp(np.multiply(-b,np.power(x,2.0))))

    def psLorentz(self,x,H):
        a = 2/(np.pi*H)
        b = 4.0/(np.power(H,2.0))
        return np.divide(a,(1+np.multiply(b,np.power(x,2.0))))

    def psEta(self,Eta0,X,_2theta):
        return Eta0+np.mulitply(X,_2theta)

    def psH(self,U,V,W,Ig,_2theta):
        #theta = np.divide(_2theta,2.0)
        theta = _2theta
        rad = np.pi/180.0
        #+np.divide(Ig,(np.power(np.cos(theta*rad),2.0)))
        return np.power(np.multiply(U,np.power(np.tan(theta*rad),2.0))+np.multiply(V,np.tan(rad*theta))+W+np.divide(Ig,(np.power(np.cos(theta*rad),2.0))),0.5)
        
    def psPseudoVoigt(self,_2theta0,_2theta,H,Eta):
        theta = _2theta-_2theta0
        return np.add(np.multiply(Eta,self.psLorentz(theta,H)),np.multiply((1-Eta),self.psGaussian(theta,H)))
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

    def writeHKL(self):
        print(self.grid.shape)
        print(self.sf.shape)
        sf = self.sf.reshape(self.sf.shape[0]*self.sf.shape[1]*self.sf.shape[2],1)
        sfComplex = self.sfComplex.reshape(self.sfComplex.shape[0]*self.sfComplex.shape[1]*self.sfComplex.shape[2],1)
        print(self.sfComplex.shape)
        #print(self.LP.shape)
        out = open("out.hkl","w")
        out.write("#  h    k    l     F()     F(imag)     |F|        d       2theta       LP   \n")
        for i in range(0,self.grid.shape[0]):
            out.write("  "+str('{:3d}'.format(int(self.grid[i][0])))+"  "+str('{:3d}'.format(int(self.grid[i][1])))+"  "+str('{:3d}'.format(int(self.grid[i][2]))))
            out.write("  "+str('{:06f}'.format(sfComplex[i][0].real)))
            out.write("  "+str('{:06f}'.format(sfComplex[i][0].imag)))
            out.write("  "+str('{:06f}'.format((sf[i][0].real))))
            out.write("  "+str('{:06f}'.format(self.d[i][0]))+"  "+str('{:06f}'.format(self.theta[i][0]*2.0))+"  "+str('{:06f}'.format(self.LP[i][0]))+"\n")

def genrateTrainingData(num,funcParams):

    dataFeat = []
    dataLabel = []

    for i in range(0,num):
        if i % 100 == 0:
            print(i)
        cell = CrystoGen()

          
        dataFeat.append(cell.pattern)
        dataLabel.append(np.array([cell.a,cell.b,cell.c,cell.alpha,cell.beta,cell.gamma,cell.sg,cell.cT]))

    dataFeat = np.array(dataFeat)
    dataLabel = np.array(dataLabel)
    print(dataFeat.shape)
    print(dataLabel.shape)
    np.save("feat.npy",dataFeat)
    np.save("label.npy",dataLabel)


#at = [['181', '0.0', '0.0', '0.0', '1.0','1.0'],['181', '-0.5', '-0.5', '0.0', '1.0','1.0'],['116','0.0','0.0','0.25','1.0','1.0'],['116','-0.209','-0.295','0.0','1.0','1.0'],['10','0.073','0.474','0.218','1.0','1.0'],['10','0.073','0.474','0.218','1.0','1.0']]
#cell = CrystoGen(Info=True)
#cell = CrystoGen(AtomTable=at,Info=True)
genrateTrainingData(100000,"")

#dd,s,a,b,c,alpha,beta,gamma = buildSF3D([-10,10],[-10,10],[-10,10],sgInfo,asfInfo)
#self.writeGRD(dd.shape,dd,"density",a,b,c,alpha,beta,gamma,Setting=0)
#self.writeGRD(ff.shape,ff,"struct",a,b,c,alpha,beta,gamma,Setting=1)

