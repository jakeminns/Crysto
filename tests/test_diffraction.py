import pandas as pd
import pytest
from .context import crysto
from pandas._testing import assert_frame_equal
import numpy as np
from unittest import mock


def test_empty_structure_factors():
    diff = crysto.Diffraction()

    sf = diff.empty_structure_factors(1)
    assert sf.shape[0] == 1

    sf = diff.empty_structure_factors(2)
    assert sf.shape[0] == 2

    assert np.sum(sf.real) == 0.0


def test_generate_reciprocal_space():
    diff = crysto.Diffraction(ReciprocalGen=[[-3, 3], [-3, 3], [-3, 3]])

    sf = diff.generate_reciprocal_space()

    assert sf.shape == (216, 3)
    assert np.min(sf[:, [0]]) == -3
    assert np.max(sf[:, [0]]) == 2
    assert np.min(sf[:, [1]]) == -3
    assert np.max(sf[:, [1]]) == 2
    assert np.min(sf[:, [2]]) == -3
    assert np.max(sf[:, [2]]) == 2


def test_calc_sf():
    diff = crysto.Diffraction(ReciprocalGen=[[-3, 3], [-3, 3], [-3, 3]])
    sf = diff.calc_sf([52, 0, 0, 0, 1, 1], np.array([[1, 1, 1]]), [0.2], False,
                      False)
    assert sf == [[1. + 0.j]]


class CubicMock:

    def __init__(self):
        self.asfInfo = [[1, 10, 1, 1, 1, 1, 1, 1, 1]]


@mock.patch('crysto.structures.Cubic', side_effect=CubicMock)
def test_calculate_atomic_scattering_factor(Cubic):
    diff = crysto.Diffraction(ReciprocalGen=[[-3, 3], [-3, 3], [-3, 3]])
    diff.structure = Cubic
    assert diff.calculate_atomic_scattering_factor(0, np.array([1])) == []
