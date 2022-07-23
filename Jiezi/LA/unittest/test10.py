import unittest

import sys

sys.path.append("../../../")

from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op
import numpy as np


class Test10(unittest.TestCase):
    vec = vector_numpy(2)
    mat = matrix_numpy(2, 2)
    sca = 2
    vec.copy([[1, 0]])
    mat.copy([[1, 1j], [1j, 1]])

    def test_multiply_and_dot(self):
        
        self.assertTrue((op.matmulvec(Test10.mat, Test10.vec.trans()).get_value() == \
                         np.array([[1], [1j]])).all(), "matmulvec is wrong")
        
        self.assertTrue((op.vecmulmat(Test10.vec, Test10.mat).get_value() == np.array([[1, 1j]])).all(), \
        "vecmulmat is wrong")
        
        self.assertTrue((op.vecmulvec(Test10.vec.trans(), Test10.vec).get_value() == \
        np.array([[1, 0], [0, 0]])).all(), "vecmulvec is wrong")
        
        self.assertEqual(op.vecdotvec(Test10.vec, Test10.vec.trans()), 1, "vecdotvec is wrong")
        
        self.assertTrue((op.matmulmat(Test10.mat, Test10.mat).get_value() == \
        np.array([[0, 2j], [2j, 0]])).all(), "matmulmat is wrong")
        
        self.assertTrue((op.matmul_sym(Test10.mat.conjugate(), Test10.mat).get_value() == \
        np.array([[2, 0], [0, 2]])).all(), "matmul_sym is wrong")
        
        self.assertTrue((op.scamulvec(Test10.sca, Test10.vec).get_value() == np.array([[2, 0]])).all(), \
        "scamulvec_row is wrong")
        self.assertTrue((op.scamulvec(Test10.sca, Test10.vec.trans()).get_value() == np.array([[2], [0]])).all(), \
        "scamulvec_column is wrong")
        
        self.assertTrue((op.scamulmat(Test10.sca, Test10.mat).get_value() == \
        np.array([[2, 2j], [2j, 2]])).all(), "scamulmat is wrong")

        self.assertTrue((op.trimatmul(Test10.mat, Test10.mat, Test10.mat).get_value() == \
                         np.array([[-2, 2j], [2j, -2]])).all(), "type nnn is wrong")

        self.assertTrue((op.trimatmul(Test10.mat, Test10.mat, Test10.mat, "cnn").get_value() == \
                         np.array([[2, 2j], [2j, 2]])).all(), "type one c is wrong")

        self.assertTrue((op.trimatmul(Test10.mat, Test10.mat, Test10.mat, "ccn").get_value() == \
                         np.array([[2, -2j], [-2j, 2]])).all(), "type two c is wrong")

        self.assertTrue((op.trimatmul(Test10.mat, Test10.mat, Test10.mat, "ccc").get_value() == \
                         np.array([[-2, -2j], [-2j, -2]])).all(), "type ccc is wrong")
