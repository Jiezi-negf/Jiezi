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
from Jiezi.LA.matrix_numpy import matrix_numpy


class TestMatrix(unittest.TestCase):
    # define variable mat
    mat = matrix_numpy(2, 3)
    ele = matrix_numpy(3, 3)
    ele.identity()
    mat_swap = matrix_numpy(3, 3)
    mat_swap.copy(np.arange(1, 10, 1).reshape(3, 3))


    def test(self):
        # make sure testing order
        self.init()
        self.get_size()
        self.set_value()
        self.set_block_value()
        self.get_value()
        self.imaginary()
        self.real()
        self.transpose()
        self.conjugate()
        self.dagger()
        self.negative()
        self.identity()
        self.trace()
        self.det()
        self.copy()
        self.eigen()
        self.swap_index()

    def init(self):
        self.assertTrue((TestMatrix.mat.get_value()
                         == np.array([[0, 0, 0], [0, 0, 0]])).all(), "init is wrong")

    def get_size(self):
        self.assertEqual(TestMatrix.mat.get_size(), (2, 3), "get_size is wrong")

    def set_value(self):
        for m in range(2):
            for n in range(3):
                TestMatrix.mat.set_value(m, n, complex(m, n + 1))
        self.assertTrue((TestMatrix.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "set_value is wrong")

    def set_block_value(self):
        tmp = np.array([[0 + 1j, 0 + 2j]])
        TestMatrix.mat.set_block_value(0, 1, 0, 2, tmp)
        self.assertTrue((TestMatrix.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "set_block_value is wrong")

    def get_value(self):
        self.assertEqual(TestMatrix.mat.get_value(1, 2), 1 + 3j, "get_value is wrong")
        self.assertTrue((TestMatrix.mat.get_value(0, 1, 1, 3) == \
                         np.array([[0 + 2j, 0 + 3j]])).all(), "get_value is wrong")

    def imaginary(self):
        self.assertTrue((TestMatrix.mat.imaginary().get_value() \
                         == np.array([[1, 2, 3], [1, 2, 3]])).all(), "imaginary is wrong")

        self.assertTrue((TestMatrix.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "original value can not be changed")

    def real(self):
        self.assertTrue((TestMatrix.mat.real().get_value() == \
                         np.array([[0, 0, 0], [1, 1, 1]])).all(), "real is wrong")

        self.assertTrue((TestMatrix.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "original value can not be changed")

    def transpose(self):
        self.assertTrue((TestMatrix.mat.trans().get_value() == \
                         np.array([[0 + 1j, 1 + 1j], [0 + 2j, 1 + 2j], [0 + 3j, 1 + 3j]])).all(), \
                        "transpose is wrong")

        self.assertTrue((TestMatrix.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "original value can not be changed")

    def conjugate(self):
        self.assertTrue((TestMatrix.mat.trans().get_value() == \
                         np.array([[0 + 1j, 1 + 1j], [0 + 2j, 1 + 2j], [0 + 3j, 1 + 3j]])).all(), \
                        "transpose is wrong")

        self.assertTrue((TestMatrix.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "original value can not be changed")

    def dagger(self):
        self.assertTrue((TestMatrix.mat.dagger().get_value() == \
                         np.array([[0 - 1j, 1 - 1j], [0 - 2j, 1 - 2j], [0 - 3j, 1 - 3j]])).all(), \
                        "dagger is wrong")

        self.assertTrue((TestMatrix.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "original value can not be changed")

    def negative(self):
        self.assertTrue((TestMatrix.mat.nega().get_value() == \
                         np.array([[0 - 1j, 0 - 2j, 0 - 3j], [-1 - 1j, -1 - 2j, -1 - 3j]])).all(), \
                        "negative is wrong")

        self.assertTrue((TestMatrix.mat.get_value() == \
                         np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
                        "original value can not be changed")

    def identity(self):

        self.assertTrue((TestMatrix.ele.get_value() == \
                         np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all(), \
                        "identity is wrong")

    def trace(self):
        self.assertEqual(TestMatrix.ele.tre(), 3, "trace is wrong")

    def det(self):
        self.assertEqual(TestMatrix.ele.det(), 1, "det is wrong")

    def copy(self):
        src = np.array([[1. + 2.j, 2. + 3.j, 2]])
        TestMatrix.mat.copy(src)

        self.assertTrue((TestMatrix.mat.get_value() == np.array([[1. + 2.j, 2. + 3.j, 2]])).all(), \
                        "copy is wrong: value")
        self.assertTrue(TestMatrix.mat.get_size() == (1, 3),
                        "copy is wrong: size")

    def eigen(self):
        a = np.diag((3, 2, 1))
        TestMatrix.mat.copy(a)
        self.assertTrue((TestMatrix.mat.eigenvalue().get_value() == [1, 2, 3]).all(), \
                        "eigenvalue is wrong")

        self.assertTrue((TestMatrix.mat.eigenvec().get_value() == \
                         np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])).all(),
                        "eigenvec is wrong")

    def swap_index(self):
        self.assertTrue((TestMatrix.mat_swap.swap_index(0, 1).get_value() == \
                         np.array([[5, 4, 6], [2, 1, 3], [8, 7, 9]])).all(),
                        "swap_index is wrong")


if __name__ == "__main__":
    unittest.main()
