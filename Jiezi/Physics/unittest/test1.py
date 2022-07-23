import unittest

import sys

sys.path.append("../../../")
from Jiezi.Physics.common import *
from Jiezi.LA.matrix_numpy import matrix_numpy


class Test1(unittest.TestCase):
    mat_ndagger = matrix_numpy(2, 2)
    mat_dagger = matrix_numpy(2, 2)
    mat_dagger.set_value(0, 0, 1)
    mat_dagger.set_value(0, 1, 1 + 1j)
    mat_dagger.set_value(1, 0, 1 - 1j)
    mat_dagger.set_value(1, 1, 0)
    mat_ndagger.copy(mat_dagger.get_value())
    mat_ndagger.set_value(0, 1, 1 - 1j)

    def test_common_func(self):
        self.assertEqual(fermi(1e-3), 0.49038580053777536)
        self.assertEqual(fermi(-1e-3), 0.50961419946222453)
        self.assertEqual(bose(1e-3), 25.50320504918597)
        self.assertEqual(bose(-1e-3), -26.503205049185993)
        self.assertEqual(heaviside(1), 1.0)
        self.assertEqual(heaviside(-1), 0.0)
        self.assertEqual(ifdagger(Test1.mat_ndagger), 4.0)
        self.assertEqual(ifdagger(Test1.mat_dagger), 0)
