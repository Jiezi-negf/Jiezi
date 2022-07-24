import unittest

import sys

sys.path.append("../../../")
from Jiezi.LA.matrix_numpy import matrix_numpy
import numpy as np


class Test4(unittest.TestCase):
    mat = matrix_numpy(2, 3)
    ele = matrix_numpy(3, 3)
    ele.identity()

    def test_matrix_compute(self):
        for m in range(2):
            for n in range(3):
                Test4.mat.set_value(m, n, complex(m, n + 1))
        self.assertTrue((Test4.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "set_value is wrong")
        tmp = np.array([[0 + 1j, 0 + 2j]])
        Test4.mat.set_block_value(0, 1, 0, 2, tmp)
        self.assertTrue((Test4.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "set_block_value is wrong")

        self.assertTrue((Test4.mat.imaginary().get_value() \
                         == np.array([[1, 2, 3], [1, 2, 3]])).all(), "imaginary is wrong")

        self.assertTrue((Test4.mat.real().get_value() == \
                         np.array([[0, 0, 0], [1, 1, 1]])).all(), "real is wrong")

        self.assertTrue((Test4.mat.trans().get_value() == \
                         np.array([[0 + 1j, 1 + 1j], [0 + 2j, 1 + 2j], [0 + 3j, 1 + 3j]])).all(), \
                        "transpose is wrong")

        self.assertTrue((Test4.mat.trans().get_value() == \
                         np.array([[0 + 1j, 1 + 1j], [0 + 2j, 1 + 2j], [0 + 3j, 1 + 3j]])).all(), \
                        "transpose is wrong")

        self.assertTrue((Test4.mat.dagger().get_value() == \
                         np.array([[0 - 1j, 1 - 1j], [0 - 2j, 1 - 2j], [0 - 3j, 1 - 3j]])).all(), \
                        "dagger is wrong")

        self.assertTrue((Test4.mat.nega().get_value() == \
                         np.array([[0 - 1j, 0 - 2j, 0 - 3j], [-1 - 1j, -1 - 2j, -1 - 3j]])).all(), \
                        "negative is wrong")

        self.assertTrue((Test4.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "original value can not be changed")

        self.assertTrue((Test4.ele.get_value() == \
                         np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all(), \
                        "identity is wrong")

        self.assertEqual(Test4.ele.tre(), 3, "trace is wrong")

        self.assertEqual(Test4.ele.det(), 1, "det is wrong")

        a = np.diag((3, 2, 1))
        Test4.mat.copy(a)
        self.assertTrue((Test4.mat.eigenvalue().get_value() == [1, 2, 3]).all(), \
                        "eigenvalue is wrong")

        self.assertTrue((Test4.mat.eigenvec().get_value() == \
                         np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])).all(),
                        "eigenvec is wrong")


