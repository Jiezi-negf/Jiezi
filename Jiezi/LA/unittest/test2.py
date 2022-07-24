import unittest

import sys

sys.path.append("../../../")
from Jiezi.LA.matrix_numpy import matrix_numpy
import numpy as np


class Test2(unittest.TestCase):
    mat = matrix_numpy(2, 3)
    ele = matrix_numpy(3, 3)
    ele.identity()

    def test_matrix_attr_set(self):
        for m in range(2):
            for n in range(3):
                Test2.mat.set_value(m, n, complex(m, n + 1))
        self.assertTrue((Test2.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "set_value is wrong")

        tmp = np.array([[0 + 1j, 0 + 2j]])
        Test2.mat.set_block_value(0, 1, 0, 2, tmp)
        self.assertTrue((Test2.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "set_block_value is wrong")