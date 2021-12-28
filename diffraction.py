import numpy as np
import decimal
from structure import *
from _structures import *


class Diffraction:
    """A class for generatating a simulated power diffraction pattern based on the passed structure parameters.

    Attributes:
        structure: Structure() object defining the structure for which a pattern will be generated.
        ReciprocalGen: Limits of reciprocal cell to generate.
    """

    def __init__(self,
                 structure,
                 ReciprocalGen=[[-15, 15], [-15, 15], [-15, 15]],
                 _lambda=0.4):

        self._lambda = _lambda
        self._reciprocal_dims = ReciprocalGen

        self._h = np.sum(np.abs(ReciprocalGen[0]))
        self._k = np.sum(np.abs(ReciprocalGen[1]))
        self._l = np.sum(np.abs(ReciprocalGen[2]))

        self.structure = structure

        self.sf, self.sfComplex, self.grid, self.d, self.theta = self.build_sf_3d(
        )

    def build_sf_3d(self,
                    FormFactor=True,
                    TempFactor=True,
                    Normalize=True,
                    RemoveF000=True):
        """Based on the atomic structure of the unit cell build the respective structure factor grid"""

        positionTable = self.structure.unitCell

        #Generate all combinations of hkl
        grid = self.generate_reciprocal_space()

        #Generate empty struture factor density
        ffComplex = self.empty_structure_factors(len(positionTable))

        d, theta = self.structure.bragg(self._lambda, grid[:, [0]],
                                        grid[:, [1]], grid[:, [2]])

        #Apply self.calc_sf across axis for all hkl's calculating all contributions of atoms for each hkl
        ffComplex = np.apply_along_axis(self.calc_sf, 1, positionTable,
                                        *[grid, theta, FormFactor, TempFactor])
        #Sum for all atoms
        ffComplex = np.sum(ffComplex, axis=0)

        if RemoveF000 == True:
            ffComplex[np.argmax(ffComplex)] = 0.0

        ffAbs = np.absolute(ffComplex)
        ffAbs = np.power(ffAbs, 2.0)

        if Normalize:
            ffAbs = ffAbs / np.amax(ffAbs)

        #Reshape for grid
        ffComplex = ffComplex.reshape(self._h, self._k, self._l, order="C")
        ffAbs = ffAbs.reshape(self._h, self._k, self._l, order="C")

        return ffAbs, ffComplex, grid, d, theta

    def empty_structure_factors(self, numAtoms):
        """Generate empty struture factor density"""
        return np.full((numAtoms, (self._h) * (self._k) * (self._l), 1),
                       0.0 + 0.0j)

    def generate_reciprocal_space(self):
        """Generate a reciprocal space array for the given dimenstions"""
        h = np.arange(self._reciprocal_dims[0][0], self._reciprocal_dims[0][1])
        k = np.arange(self._reciprocal_dims[1][0], self._reciprocal_dims[1][1])
        l = np.arange(self._reciprocal_dims[2][0], self._reciprocal_dims[2][1])
        #Generate all combinations of hkl
        return np.array(np.meshgrid(h, k, l)).T.reshape(
            h.shape[0] * k.shape[0] * l.shape[0], 3, order="C")

    def calc_sf(self, atom, grid, theta, FormFactor, TempFactor):
        """Calculate atomic scattering factor"""

        pos = atom[1:4]

        sf = np.exp(
            np.multiply(-2.0j * np.pi, (np.multiply(grid[:, [0]], pos[0]) +
                                        np.multiply(grid[:, [1]], pos[1]) +
                                        np.multiply(grid[:, [2]], pos[2]))))

        if FormFactor == True:
            asf = self.calculate_atomic_scattering_factor(int(atom[0]), theta)
            asf = asf.reshape(asf.shape[0], 1)
            sf = np.multiply(asf, sf, casting='same_kind')

        if TempFactor == True:
            isoB = self.iso_displacement_factor(theta, atom[5])
            sf = np.multiply(isoB, sf, casting="same_kind")

        #sf = np.multiply(sf,atom[4])
        return sf

    def calculate_atomic_scattering_factor(self, element, theta):
        """For the given element calculate the atomic scattering factor"""
        params = self.structure.asfInfo[element][1:9]
        rad = np.pi / 180.0

        params = params.reshape(4, 2)
        c = self.structure.asfInfo[element][9]
        theta1 = np.sin(theta * rad) / self._lambda

        expo = np.power(theta1, 2.0)
        expo = -np.outer(params[:, [1]], expo).T
        f = np.exp(expo).dot(params[:, [0]])
        f = f + c

        return f

    def iso_displacement_factor(self, theta, atomB):
        """Isothermal displacment parameter"""
        rad = np.pi / 180.0
        return np.exp(
            np.multiply(-atomB, (np.divide(np.power(np.sin(theta * rad), 2.0),
                                           np.power(self._lambda, 2.0)))))

    def lp_correction(self, theta):
        """Lorentz-polarization factor"""
        rad = np.pi / 180.0
        return np.divide((1 + np.cos(2 * rad * theta)**2),
                         np.multiply(np.cos(theta * rad),
                                     np.sin(theta * rad)**2))

    def convert_to_position(self, positionTable, a, b, c, alpha, beta, gamma):

        rad = np.pi / 180.0
        omega = a * b * c * (1 - (np.cos(alpha * rad)**2) -
                             (np.cos(beta * rad)**2) -
                             (np.cos(gamma * rad)**2) + 2 * np.cos(alpha * rad)
                             * np.cos(beta * rad) * np.cos(gamma * rad))**0.5
        convertArr = np.array(
            [[a, b * np.cos(gamma * rad), c * np.cos(beta * rad)],
             [
                 0, b * np.sin(gamma * rad),
                 c * ((np.cos(beta * rad) * np.cos(gamma * rad) -
                       np.cos(gamma * rad)) / (np.sin(gamma * rad)))
             ], [0.0, 0.0, omega / (a * b * np.sin(gamma * rad))]])
        out = []
        for i in positionTable:
            out.append(np.matmul(np.array(i).transpose(), convertArr).tolist())

        return out

    def calculate_powder_pattern(self,
                                 _2thetalim,
                                 incr,
                                 U,
                                 V,
                                 W,
                                 Ig,
                                 Eta0,
                                 X,
                                 width,
                                 ScaleFactor,
                                 plot=True):
        """Generate Simulated Powder Diffraction Pattern"""
        sf = self.sf.reshape(
            self.sf.shape[0] * self.sf.shape[1] * self.sf.shape[2], 1)

        #Empty Pattern arrays
        pattern = np.ones((int(_2thetalim / incr), 1))

        _2theta = self.theta.real * 2.0

        #Convert width from 2theta to positions in array
        widthPat = int(width / incr)

        for i in range(0, _2theta.shape[0]):
            if round(sf[i][0].real, 3) != 0:
                #Locate position (item number) of 2theta in pattern array
                pos = int(np.nan_to_num(np.round(_2theta[i][0] / incr)))

                if pos < pattern.shape[0]:
                    _2thetaZero = _2theta[i][0]

                    #max min pattern values for all 2theta values between width
                    patternBoundaryMin = pos - widthPat
                    patternBoundaryMax = pos + widthPat

                    _2thetaSelection = np.linspace(
                        _2thetaZero - width, _2thetaZero + width,
                        patternBoundaryMax - patternBoundaryMin)
                    _2thetaSelection = _2thetaSelection.reshape(
                        _2thetaSelection.shape[0], 1)

                    H = self.h_peak_shape(U, V, W, Ig, _2thetaZero)
                    ps = self.pseudo_voigt_peak_shape(_2thetaZero,
                                                      _2thetaSelection, H,
                                                      Eta0)

                    newInt = np.multiply((sf[i][0].real), ps)

                    if patternBoundaryMin < 0:
                        newInt = newInt[np.abs(patternBoundaryMin):]
                        patternBoundaryMin = 0

                    if patternBoundaryMax > pattern.shape[0]:
                        patternBoundaryMax = pattern.shape[0]
                        newInt = newInt[0:int(patternBoundaryMax -
                                              patternBoundaryMin)]

                    if pattern[patternBoundaryMin:patternBoundaryMax].shape[
                            0] != 0 or pattern.shape[0] > patternBoundaryMin:
                        pattern[
                            patternBoundaryMin:patternBoundaryMax] = np.add(
                                pattern[patternBoundaryMin:patternBoundaryMax],
                                newInt)

        pattern = pattern / np.amax(pattern)
        pattern = np.multiply(pattern, ScaleFactor)

        _2thetaAxis = np.arange(0, _2thetalim, incr)
        _2thetaAxisRe = _2thetaAxis.reshape(_2thetaAxis.shape[0], 1)[30:]

        #LP_ = self.lp_correction(_2thetaAxisRe/2.0)
        #LP_[np.argmax(LP_)] = 0.0
        #LP_[:500] = 1.0
        #pattern = np.multiply(LP_,pattern[30:])

        if plot:
            import matplotlib.pyplot as plt
            plt.plot(_2thetaAxis[:3000], pattern[:3000], "r")
            plt.xlabel("2theta")
            plt.ylabel("Intensity (Normalised)")
            plt.show()

        return _2thetaAxis[:3000], pattern[:3000]

    def gaussian_peak_shape(self, x, H):
        """Gaussian peak shape"""
        a = (1.0 / H) * (4.0 * np.log(2.0) / np.pi)**0.5
        b = (4 * np.log(2.0)) / (H**2.0)
        return np.multiply(a, np.exp(np.multiply(-b, np.power(x, 2.0))))

    def lorentz_peak_shape(self, x, H):
        """Lorentz peak shape"""
        a = 2 / (np.pi * H)
        b = 4.0 / (np.power(H, 2.0))
        return np.divide(a, (1 + np.multiply(b, np.power(x, 2.0))))

    def eta_peak_shape(self, Eta0, X, _2theta):
        """eta peak shape"""
        return Eta0 + np.mulitply(X, _2theta)

    def h_peak_shape(self, U, V, W, Ig, _2theta):
        """h peak shape"""
        theta = _2theta
        rad = np.pi / 180.0
        return np.power(
            np.multiply(U, np.power(np.tan(theta * rad), 2.0)) +
            np.multiply(V, np.tan(rad * theta)) + W +
            np.divide(Ig, (np.power(np.cos(theta * rad), 2.0))), 0.5)

    def pseudo_voigt_peak_shape(self, _2theta0, _2theta, H, Eta):
        """pseudo voight peak shape"""
        theta = _2theta - _2theta0
        return np.add(
            np.multiply(Eta, self.lorentz_peak_shape(theta, H)),
            np.multiply((1 - Eta), self.gaussian_peak_shape(theta, H)))

    def SF_to_struct(self, sf):
        """Scattering factor to structure factor conversion"""
        sf = np.fft.fftshift(sf)
        sf = np.fft.fftn(sf).real1
        return (sf)

    def writePowderPlot(self, axis, intensity):
        """write plot data to file"""
        file_ = open("data/plot.dat", "w")
        for theta, int_ in zip(axis, intensity):
            file_.write(
                str('{:06f}'.format(theta)) + "   " +
                str('{:06f}'.format(int_[0])) + "\n")

    def writeGRD(self,
                 fdat,
                 data,
                 name,
                 a,
                 b,
                 c,
                 alpha,
                 beta,
                 gamma,
                 Setting=0):
        """Write reciprocal grid to file"""
        out = open(name + ".grd", "w")
        out.write("Title: Put your title \n")
        if Setting == 0:
            out.write(" " + str(fdat[0]) + "  " + str(fdat[1]) + "  " +
                      str(fdat[2]) + "  " + str(90) + "  " + str(90) + "  " +
                      str(90) + "\n")
        else:
            out.write(" " + str(a) + "  " + str(b) + "  " + str(c) + "  " +
                      str(alpha) + "  " + str(beta) + "  " + str(gamma) + "\n")

        cell = np.array(fdat) / 10.0

        out.write("   " + str(fdat[0]) + "    " + str(fdat[1]) + "   " +
                  str(fdat[2]) + " \n")

        for x in range(0, (fdat[0])):
            for y in range(0, (fdat[1])):
                for z in range(0, (fdat[2])):
                    out.write('%.6E' % decimal.Decimal(data[x][y][z]) + "   ")
                    if z % 6 == 0:
                        out.write("\n")
                out.write("\n")


import pandas as pd
import zlib
import pickle


def pickle_file(input_object):
    return pickle.dumps(input_object, protocol=-1)


def decompress(obj):
    return pickle.loads(zlib.decompress(obj))


def compress(input_object):
    return zlib.compress(pickle.dumps(input_object), level=1)


def genrateTrainingData(size):

    from tqdm import tqdm

    for num in range(8, 15):
        dataFeat = []
        dataLabel = []
        for i in tqdm(range(0, size)):

            for ct in [
                    Cubic, Monoclinic, Orthorhombic, Tetragonal, Trigonal,
                    Hexagonal
            ]:
                if i % 100 == 0:
                    print(i)

                cell = Struct().create(ct())
                pattern = Diffraction(cell)
                _theta, pattern = pattern.calculate_powder_pattern(180.0,
                                                                   0.01,
                                                                   0.1,
                                                                   -0.0001,
                                                                   0.0001,
                                                                   0.001,
                                                                   0.5,
                                                                   0.001,
                                                                   8.0,
                                                                   1.0,
                                                                   plot=False)

                dataFeat.append(pattern.reshape(-1))
                dataLabel.append(
                    np.array([
                        cell.a, cell.b, cell.c, cell.alpha, cell.beta,
                        cell.gamma, cell.sg, cell.cT,
                        len(cell.atomTable)
                    ]))

        dataFeat = np.array(dataFeat)
        dataLabel = np.array(dataLabel)

        print(dataFeat.shape)
        print(dataLabel.shape)

        df_feat = pd.DataFrame(dataFeat, columns=_theta)
        obj = compress(df_feat)

        with open('feat_new_{}.pklz'.format(str(num)), 'wb') as f:
            f.write(obj)

        df_labels = pd.DataFrame(dataLabel,
                                 columns=[
                                     'a', 'b', 'c', 'alpha', 'beta', 'gamma',
                                     'sg', 'cT', 'atoms'
                                 ])
        obj = compress(df_labels)

        with open('labels_new_{}.pklz'.format(str(num)), 'wb') as f:
            f.write(obj)

    #np.save("feat.npy",dataFeat)
    #np.save("label.npy",dataLabel)


"""
##### Random ####
cell = Struct().create(Cubic())
pattern = PatternGen(cell)
pattern = pattern.calculate_powder_pattern(180.0,0.01,0.1,-0.0001,0.0001,0.001,0.5,0.001,8.0,1.0)



##### Cell #####
at = [['62', '0.0', '0.0', '0.0', '1.0', '1.0']]
cell = Struct(random=False).create(Hexagonal(), a=3, c=6, sg=167, atomTable=at)
pattern = Diffraction(cell)
pattern = pattern.calculate_powder_pattern(180.0, 0.01, 0.1, -0.0001, 0.0001,
                                           0.001, 0.5, 0.001, 8.0, 1.0)
"""
#### Data Gen ####
#genrateTrainingData(150)
