import pandas as pd
import pytest
from .context import crysto
from pandas._testing import assert_frame_equal
import numpy as np


def test_empty_structure_factors():
    diff = crysto.Diffraction()
    sf = diff.empty_structure_factors(1)
    print(sf)
