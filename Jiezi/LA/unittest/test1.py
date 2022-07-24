import unittest

import sys

sys.path.append("../../../")
from Jiezi.LA.matrix_numpy import matrix_numpy
import numpy as np


class Test1(unittest.TestCase):
    mat = matrix_numpy(2, 3)
    ele = matrix_numpy(3, 3)
    ele.identity()

    def test_matrix_init(self):
        self.assertTrue((Test1.mat.get_value()
                         == np.array([[0, 0, 0], [0, 0, 0]])).all(), "init is wrong")
