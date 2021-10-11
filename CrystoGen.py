import numpy as np
import sys
import decimal

class CrystoGen:
    """A class for generatating a random crystal structure.

    Attributes:
        A, B, C, alpha,beta and gamma: Crystallographic unit cell parameters.
        spaceGroup: Space Group number according to International Tables for Crystallography.
        atomTable: A list describing the atoms in the unit cell.
        info: Specify if structure info is to be printed.
        reciprocalGen: Limits of reciprocal cell to generate.
    """

    def __init__(self,A=None,B=None,C=None,alpha=None,beta=None,gamma=None,spaceGroup=None,crystalSystem=None,atomTable=None,info=False):
        
        self.a,self.b,self.c,self.alpha,self.beta,self.gamma,self.sg,self.cT=self.randomCell(A,B,C,alpha,beta,gamma,spaceGroup,crystalSystem)
        self.atomTable = self.randomAtoms(atomTable)
        self.sgInfo = self.readSGLib()
        self.asfInfo = self.readASF()

        if info:
            self.info()
            
    def info(self):
        print("========== Crystal Info ==========")
        print("")
        print("a: ",self.a)
        print("b: ",self.b)
        print("c: ",self.c)
        print("alpha: ",self.alpha)
        print("beta: ",self.beta)
        print("gamma: ",self.gamma)
        print("Space Group No.:{}, Laue Class:{}".format(self.sg+1,self.sgInfo[self.sg][2][0],self.sgInfo[self.sg][1][0]))
        print("")
        print("-------- Atom Table --------")
        
        for atom in self.atomTable:
            print(atom)
        print("")
        print("=================================")

    def randomAtoms(self,atomTable,size=4):

        if atomTable == None:

            atomTable = []

            for atom in range(0,np.random.randint(1,size)):
                atomTable.append(["10",str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),str(np.random.random_sample(1)[0]),"1.0","1.0"])

            return atomTable

        else:
            return atomTable

    def randomCell(self,A,B,C,alpha,beta,gamma,sG,cT):

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
        if alpha != None:
            alpha = alpha
        if beta != None:
            beta = beta
        if gamma != None:
            gamma = gamma

        return a,b,c,alpha,beta,gamma,sg,cT

    def randomCellAngle(self):
        return np.random.uniform(90.5,120.0)

    def randomCellParam(self):
        return np.random.uniform(0.5,30.0)

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
