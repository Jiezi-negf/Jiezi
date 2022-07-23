import unittest

import sys

sys.path.append("../../../")

from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op
import numpy as np


class Test9(unittest.TestCase):
    vec = vector_numpy(2)
    mat = matrix_numpy(2, 2)
    sca = 2
    vec.copy([[1, 0]])
    mat.copy([[1, 1j], [1j, 1]])

    def test_plus_and_minus(self):
        self.assertTrue((op.addmat(Test9.mat, Test9.mat, Test9.mat).get_value() == \
                         np.array([[3, 3j], [3j, 3]])).all(), "addmat is wrong")
        self.assertTrue((op.addvec(Test9.vec, Test9.vec, Test9.vec, Test9.vec).get_value() == \
                         np.array([[4, 0]])).all(), "addvec is wrong")