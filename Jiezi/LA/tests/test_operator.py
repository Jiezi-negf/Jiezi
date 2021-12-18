# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys
import numpy as np
import unittest

sys.path.append("../../../")

from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op

class TestOp(unittest.TestCase):

    def test(self):
        
        vec = vector_numpy(2)
        mat = matrix_numpy(2, 2)
        sca = 2
        vec.copy([1, 0])
        mat.copy([[1, 1j], [1j, 1]])

        # test matmulvec
        self.assertTrue((op.matmulvec(mat, vec.trans()).get_value() == \
        np.array([[1], [1j]])).all(), "matmulvec is wrong")

        # test vecmulmat
        self.assertTrue((op.vecmulmat(vec, mat).get_value() == np.array([[1, 1j]])).all(), \
        "vecmulmat is wrong")

        # test vecmulvec
        self.assertTrue((op.vecmulvec(vec.trans(), vec).get_value() == \
        np.array([[1, 0], [0, 0]])).all(), "vecmulvec is wrong")

        # test vecdotvec
        self.assertEqual(op.vecdotvec(vec, vec.trans()), 1, "vecdotvec is wrong")

        # test matmulmat
        self.assertTrue((op.matmulmat(mat, mat).get_value() == \
        np.array([[0, 2j], [2j, 0]])).all(), "matmulmat is wrong")

        # test matmul_sym
        self.assertTrue((op.matmul_sym(mat.conjugate(), mat).get_value() == \
        np.array([[2, 0], [0, 2]])).all(), "matmul_sym is wrong")

        # test scamulvec
        self.assertTrue((op.scamulvec(sca, vec).get_value() == np.array([2, 0])).all(), \
        "scamulvec is wrong")

        # test scamulmat
        self.assertTrue((op.scamulmat(sca, mat).get_value() == \
        np.array([[2, 2j], [2j, 2]])).all(), "scamulmat is wrong")

        # test trimatmul
        self.assertTrue((op.trimatmul(mat, mat, mat).get_value() == \
        np.array([[-2, 2j], [2j, -2]])).all(), "type nnn is wrong")

        self.assertTrue((op.trimatmul(mat, mat, mat, "cnn").get_value() == \
        np.array([[2, 2j], [2j, 2]])).all(), "type one c is wrong")

        self.assertTrue((op.trimatmul(mat, mat, mat, "ccn").get_value() == \
        np.array([[2, -2j], [-2j, 2]])).all(), "type two c is wrong")
            
        self.assertTrue((op.trimatmul(mat, mat, mat, "ccc").get_value() == \
        np.array([[-2, -2j], [-2j, -2]])).all(), "type ccc is wrong")

        # test addmat
        self.assertTrue((op.addmat(mat, mat, mat).get_value() == \
        np.array([[3, 3j], [3j, 3]])).all(), "addmat is wrong")

        # test addvec
        self.assertTrue((op.addvec(vec, vec, vec, vec).get_value() == \
        np.array([4, 0])).all(), "addvec is wrong")

        # test inv
        mat.copy(np.array([[1, 2], [0, 1]]))
        self.assertTrue((op.inv(mat).get_value() == np.array([[1, -2], [0, 1]])).all(), \
        "inv is wrong")



if __name__ == "__main__":
    unittest.main()


