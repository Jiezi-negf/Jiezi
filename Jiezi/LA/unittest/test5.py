import unittest

import sys

sys.path.append("../../../")
from Jiezi.LA.vector_numpy import vector_numpy
import numpy as np


class Test5(unittest.TestCase):
    vec = vector_numpy(3)

    def test_vector_init(self):
        self.assertTrue((Test5.vec.get_value() == \
                         np.array([[0 + 0j], [0 + 0j], [0 + 0j]])).all(), \
                        "init is wrong")
