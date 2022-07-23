import unittest

import sys

sys.path.append("../../../")
from Jiezi.LA.vector_numpy import vector_numpy
import numpy as np


class Test8(unittest.TestCase):
    vec = vector_numpy(3)

    def test_vector_compute(self):
        Test8.vec.set_value((0, 0), 1 + 3j)
        Test8.vec.set_value((1, 0), 2 + 2j)
        self.assertTrue((Test8.vec.get_value() == np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), \
                        "set_value is wrong")

        self.assertTrue((Test8.vec.imaginary().get_value() == np.array([[3], [2], [0]])).all(), \
                        "imaginary is wrong")
        self.assertTrue((Test8.vec.real().get_value() == \
                         np.array([[1], [2], [0]])).all(), "real is wrong")
        self.assertTrue((Test8.vec.trans().get_value() == \
                         np.array([[1 + 3j, 2 + 2j, 0 + 0j]])).all(), "transpose is wrong")
        self.assertTrue((Test8.vec.conjugate().get_value() == \
                         np.array([[1 - 3j], [2 - 2j], [0 - 0j]])).all(), "conjugate is wrong")
        self.assertTrue((Test8.vec.dagger().get_value() == \
                         np.array([[1 - 3j], [2 - 2j], [0 - 0j]])).all(), "dagger is wrong")
        self.assertTrue((Test8.vec.nega().get_value() == \
                         np.array([[-1 - 3j], [-2 - 2j], [0]])).all(), "negative is wrong")
        self.assertTrue((Test8.vec.get_value() == \
                         np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), "original value can not be changed")