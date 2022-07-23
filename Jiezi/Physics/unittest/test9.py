import unittest

import sys

sys.path.append("../../../")

from Jiezi.FEM.shape_function import shape_function
import numpy as np


class Test9(unittest.TestCase):

    def test_shape_func(self):
        coord = [1, 2, 1]
        dof_coor = [[4, 5, 6], [2, 7, 1], [5, 2, 3], [1, 3, 7]]
        co_shapefunc = shape_function(dof_coor)
        matrix_1 = np.arange(16).reshape(4, 4)
        for i in range(4):
            matrix_1[i, :] = [1] + dof_coor[i]
        matrix_2 = np.array(co_shapefunc)
        res = np.dot(matrix_2, matrix_1)
        ref = np.eye(4)
        error = 0.0
        for i in range(4):
            for j in range(4):
                error += abs(ref[i, j] - res[i, j])
        self.assertGreater(1e-10, error)
