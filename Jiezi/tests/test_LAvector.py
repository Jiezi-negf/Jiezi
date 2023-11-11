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


class TestVector(unittest.TestCase):
    vec = vector_numpy(3)

    def test(self):
        self.init()
        self.get_size()
        self.set_value()
        self.get_value()
        self.imaginary()
        self.real()
        self.transpose()
        self.conjugate()
        self.dagger()
        self.negative()

    def init(self):
        self.assertTrue((TestVector.vec.get_value() == \
                         np.array([[0 + 0j], [0 + 0j], [0 + 0j]])).all(), \
                        "init is wrong")

    def get_size(self):
        self.assertEqual(TestVector.vec.get_size(), (3, 1), "get_size is wrong")

    def set_value(self):
        TestVector.vec.set_value((0, 0), 1 + 3j)
        TestVector.vec.set_value((1, 0), 2 + 2j)
        self.assertTrue((TestVector.vec.get_value() == np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), \
                        "set_value is wrong")

    def get_value(self):
        self.assertEqual(TestVector.vec.get_value(2), 0 + 0j, \
                         "get_value is wrong")

        self.assertTrue((TestVector.vec.get_value(0, 2) == np.array([[1 + 3j], [2 + 2j]])).all(), \
                        "get_value is wrong")

    def imaginary(self):
        self.assertTrue((TestVector.vec.imaginary().get_value() == np.array([[3], [2], [0]])).all(), \
                        "imaginary is wrong")

        self.assertTrue((TestVector.vec.get_value() == np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), \
                        "original value can not be changed")

    def real(self):
        self.assertTrue((TestVector.vec.real().get_value() == \
                         np.array([[1], [2], [0]])).all(), "real is wrong")

        self.assertTrue((TestVector.vec.get_value() == \
                         np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), \
                        "original value can not be changed")

    def transpose(self):
        # print(TestVector.vec.trans().get_value())
        self.assertTrue((TestVector.vec.trans().get_value() == \
                         np.array([[1 + 3j, 2 + 2j, 0 + 0j]])).all(), "transpose is wrong")

        self.assertTrue((TestVector.vec.get_value() == np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), \
                        "original value can not be changed")

    def conjugate(self):
        self.assertTrue((TestVector.vec.conjugate().get_value() == \
                         np.array([[1 - 3j], [2 - 2j], [0 - 0j]])).all(), "conjugate is wrong")

        self.assertTrue((TestVector.vec.get_value() == \
                         np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), "original value can not be changed")

    def dagger(self):
        self.assertTrue((TestVector.vec.dagger().get_value() == \
                         np.array([[1 - 3j], [2 - 2j], [0 - 0j]])).all(), "dagger is wrong")

        self.assertTrue((TestVector.vec.get_value() == \
                         np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), "original value can not be changed")

    def negative(self):
        self.assertTrue((TestVector.vec.nega().get_value() == \
                         np.array([[-1 - 3j], [-2 - 2j], [0]])).all(), "negative is wrong")

        self.assertTrue((TestVector.vec.get_value() == \
                         np.array([[1 + 3j], [2 + 2j], [0 + 0j]])).all(), "original value can not be changed")

    def copy(self):
        src = np.array([[1. + 2.j, 2. + 3.j, 2]])
        TestVector.vec.copy(src)
        self.assertTrue((TestVector.vec.get_value() == np.array([[1. + 2.j, 2. + 3.j, 2]])).all(),
                        "copy is wrong: value")
        self.assertTrue(TestVector.vec.get_size() == (1, 3),
                        "copy is wrong: size")


if __name__ == "__main__":
    unittest.main()


