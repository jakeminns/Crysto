import numpy as np


class IStruct:
    """An interface class for generatating random crystal structures.

    Attributes:
        a, b, c, alpha, beta and gamma: Crystallographic unit cell parameters.
        cT: Crystal type.
        sg: Space Group number according to International Tables for Crystallography.
        atomTable: A list describing the atoms in the unit cell.
        unitCell: A list describing the atoms in the unit cell after application of symmetry and centering operations.
        sgInfo: Specify if structure info is to be printed.
    """

    def __init__(self, ct):

        self.sgInfo = self.readSGLib()
        self.asfInfo = self.readASF()

        self.a = None
        self.b = None
        self.c = None
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.sg = None
        self.cT = ct
        self.atomTable = None
        self.unitCell = None

    def addAtoms(self, random=True, atomTable=None, max_size=4, min_size=2):
        """Generate a random set of atoms if None are supplied"""
        if random:
            atomTable = []
            for i in range(0, np.random.randint(min_size, max_size)):
                atomTable.append([
                    "10",
                    str(np.random.random_sample(1)[0]),
                    str(np.random.random_sample(1)[0]),
                    str(np.random.random_sample(1)[0]),
                    "1.0",
                    "1.0",
                ])

        self.atomTable = atomTable

    def randomCellAngle(self):
        """Random angle"""
        return np.random.uniform(90.5, 120.0)

    def randomCellParam(self):
        """Random cell params"""
        return np.random.uniform(3, 30.0)

    def readSGLib(self):
        """Read and parse sym data sheet"""
        file_ = open("data/syminfo.txt", "r")
        lines = file_.read().split("\n")
        lines = list(filter(None, lines))
        index = []
        sgData = []
        for i in range(0, len(lines)):
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
                sym.append(line.split()[1].split(","))
            elif line.split()[0] == "cenop":
                cen.append(line.split()[1].split(","))
            elif line.split()[0] == "symbol" and line.split()[1] == "laue":
                laue.append(line.split("'")[3])
            elif line.split()[0] == "symbol" and line.split()[1] == "xHM":
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
        """Read atomic scattering factor data sheet"""
        file_ = np.genfromtxt("data/atf.txt", unpack=True).T
        return file_

    def build_apply_symmetry_opperations(self, positions, symmetryOpp):
        """Apply a set of symmerty operations to each position"""
        NewPositions = []
        for sym in range(0, len(symmetryOpp)):
            for pos in range(0, len(positions)):
                tempPos = [
                    positions[pos][0],  # Atom Type
                    self.apply_opperation(positions[pos],
                                          symmetryOpp[sym][0]),  # X
                    self.apply_opperation(positions[pos],
                                          symmetryOpp[sym][1]),  # Y
                    self.apply_opperation(positions[pos],
                                          symmetryOpp[sym][2]),  # Z
                    positions[pos][4],  # Occ
                    positions[pos][5],
                ]  # Iso
                NewPositions.append(tempPos)
        return NewPositions

    def clean_atom_table(self, table):
        """Clean the unit cell table."""
        # Convert to numpy array
        table = np.array(table, dtype=np.dtype(float))
        # Modulo Positions
        table[:, [1, 2, 3]] = table[:, [1, 2, 3]] % 1
        # Find absolute
        table = np.abs(table)
        # Round results for removal of duplicates
        table = np.around(table, decimals=6)
        # Remove duplicates
        table = np.unique(table, axis=0)
        return table

    def getCellOps(self):
        """Get the centering and symmetery operations relevent to the structures space group."""
        return self.sgInfo[self.sg][3:][0], self.sgInfo[self.sg][4:][0]

    def remove_brackets(self, input):
        """Remove brackets from input string"""
        return str(input).replace("(", "").replace(")", "")

    def apply_opperation(self, position, symmetryOpp):
        """Evaluate the symmetry operation on the position (xyz)"""
        a = symmetryOpp.replace("x", str(self.remove_brackets(position[1])))
        a = a.replace("y", str(self.remove_brackets(position[2])))
        a = a.replace("z", str(self.remove_brackets(position[3])))
        return round(eval(a), 3)

    def buildUnitCell(self):
        """Apply space group centering and symmetry operations to atoms in unit cell."""
        sym, cen = self.getCellOps()
        positionTable = self.build_apply_symmetry_opperations(
            self.atomTable, cen)
        positionTable = self.build_apply_symmetry_opperations(
            positionTable, sym)
        self.unitCell = self.clean_atom_table(positionTable)

    def bragg(self, _lambda, h, k, l):
        """Perform the quadratic bragg equation for each crystal type"""
        hkl = self.cell_bragg(h, k, l)
        d = np.power(np.divide(1.0, hkl), 0.5)
        return d, (180.0 / np.pi) * np.divide(_lambda, np.multiply(2.0, d))
