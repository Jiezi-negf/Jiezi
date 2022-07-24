import unittest

import sys

sys.path.append("../../../")
from Jiezi.LA.vector_numpy import vector_numpy
import numpy as np


class Test7(unittest.TestCase):
    vec = vector_numpy(3)

    def test_vector_attr_get(self):
        Test7.vec.set_value((0, 0), 1 + 3j)
        Test7.vec.set_value((1, 0), 2 + 2j)
        self.assertTrue((Test7.vec.get_value() == np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), \
                        "set_value is wrong")
        self.assertEqual(Test7.vec.get_size(), (3, 1), "get_size is wrong")
        self.assertEqual(Test7.vec.get_value(2), 0 + 0j, \
                         "get_value is wrong")
        self.assertTrue((Test7.vec.get_value(0, 2) == np.array([[1 + 3j], [2 + 2j]])).all(), \
                        "get_value is wrong")
