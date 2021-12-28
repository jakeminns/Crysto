import pandas as pd
import zlib
import pickle
import numpy as np
from crysto import *


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
genrateTrainingData(150)
