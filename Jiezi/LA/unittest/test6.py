import unittest

import sys

sys.path.append("../../../")
from Jiezi.LA.vector_numpy import vector_numpy
import numpy as np


class Test6(unittest.TestCase):
    vec = vector_numpy(3)

    def test_vector_attr_set(self):
        Test6.vec.set_value((0, 0), 1 + 3j)
        Test6.vec.set_value((1, 0), 2 + 2j)
        self.assertTrue((Test6.vec.get_value() == np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), \
                        "set_value is wrong")
