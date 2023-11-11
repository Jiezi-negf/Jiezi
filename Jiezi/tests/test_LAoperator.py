# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import os
import sys
import numpy as np
import unittest

sys.path.append(os.path.abspath(__file__ + "/../../.."))

from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op


class TestOp(unittest.TestCase):
    vec = vector_numpy(2)
    mat = matrix_numpy(2, 2)
    sca = 2
    vec.copy([[1, 0]])
    mat.copy([[1, 1j], [1j, 1]])

    def test(self):
        self.matmulvec()
        self.vecmulmat()
        self.vecdotvec()
        self.matmulmat()
        self.matmul_sym()
        self.scamulvec()
        self.scamulmat()
        self.trimatmul()
        self.addmat()
        self.addvec()
        self.inv()
        self.qr_decomp()

    # test matmulvec
    def matmulvec(self):
        self.assertTrue((op.matmulvec(TestOp.mat, TestOp.vec.trans()).get_value() == \
                         np.array([[1], [1j]])).all(), "matmulvec is wrong")

    # test vecmulmat
    def vecmulmat(self):
        self.assertTrue((op.vecmulmat(TestOp.vec, TestOp.mat).get_value() == np.array([[1, 1j]])).all(), \
                        "vecmulmat is wrong")

    # test vecmulvec
    def vecmulvec(self):
        self.assertTrue((op.vecmulvec(TestOp.vec.trans(), TestOp.vec).get_value() == \
                         np.array([[1, 0], [0, 0]])).all(), "vecmulvec is wrong")

    # test vecdotvec
    def vecdotvec(self):
        self.assertEqual(op.vecdotvec(TestOp.vec, TestOp.vec.trans()), 1, "vecdotvec is wrong")

    # test matmulmat
    def matmulmat(self):
        self.assertTrue((op.matmulmat(TestOp.mat, TestOp.mat).get_value() == \
                         np.array([[0, 2j], [2j, 0]])).all(), "matmulmat is wrong")

    # test matmul_sym
    def matmul_sym(self):
        self.assertTrue((op.matmul_sym(TestOp.mat.conjugate(), TestOp.mat).get_value() == \
                         np.array([[2, 0], [0, 2]])).all(), "matmul_sym is wrong")

    # test scamulvec
    def scamulvec(self):
        self.assertTrue((op.scamulvec(TestOp.sca, TestOp.vec).get_value() == np.array([[2, 0]])).all(), \
                        "scamulvec_row is wrong")
        self.assertTrue((op.scamulvec(TestOp.sca, TestOp.vec.trans()).get_value() == np.array([[2], [0]])).all(), \
                        "scamulvec_column is wrong")

    # test scamulmat
    def scamulmat(self):
        self.assertTrue((op.scamulmat(TestOp.sca, TestOp.mat).get_value() == \
                         np.array([[2, 2j], [2j, 2]])).all(), "scamulmat is wrong")

    # test trimatmul
    def trimatmul(self):
        self.assertTrue((op.trimatmul(TestOp.mat, TestOp.mat, TestOp.mat).get_value() == \
                         np.array([[-2, 2j], [2j, -2]])).all(), "type nnn is wrong")

        self.assertTrue((op.trimatmul(TestOp.mat, TestOp.mat, TestOp.mat, "cnn").get_value() == \
                         np.array([[2, 2j], [2j, 2]])).all(), "type one c is wrong")

        self.assertTrue((op.trimatmul(TestOp.mat, TestOp.mat, TestOp.mat, "ccn").get_value() == \
                         np.array([[2, -2j], [-2j, 2]])).all(), "type two c is wrong")

        self.assertTrue((op.trimatmul(TestOp.mat, TestOp.mat, TestOp.mat, "ccc").get_value() == \
                         np.array([[-2, -2j], [-2j, -2]])).all(), "type ccc is wrong")

    # test addmat
    def addmat(self):
        self.assertTrue((op.addmat(TestOp.mat, TestOp.mat, TestOp.mat).get_value() == \
                         np.array([[3, 3j], [3j, 3]])).all(), "addmat is wrong")

    # test addvec
    def addvec(self):
        self.assertTrue((op.addvec(TestOp.vec, TestOp.vec, TestOp.vec, TestOp.vec).get_value() == \
                         np.array([[4, 0]])).all(), "addvec is wrong")

    # test inv
    def inv(self):
        TestOp.mat.copy(np.array([[1, 2], [0, 1]]))
        self.assertTrue((op.inv(TestOp.mat).get_value() == np.array([[1, -2], [0, 1]])).all(), \
                        "inv is wrong")

    # test qr decomposition
    def qr_decomp(self):
        TestOp.mat.copy(np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]]))
        self.assertTrue(np.linalg.norm(op.inv(TestOp.mat).get_value() -
                                       np.array([[-4.08248290e-01, 5.77350269e-01, -7.07106781e-01],
                                                 [-8.16496581e-01, -5.77350269e-01, -3.33066907e-16],
                                                 [-4.08248290e-01, 5.77350269e-01, 7.07106781e-01]]) < 1e-6),
                        "qr decomposition is wrong")


if __name__ == "__main__":
    unittest.main()


