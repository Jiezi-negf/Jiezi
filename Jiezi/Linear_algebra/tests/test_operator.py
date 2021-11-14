# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys
import os
import numpy as np
from Jiezi.Linear_algebra.vector_numpy import vector_numpy
from Jiezi.Linear_algebra.matrix_numpy import matrix_numpy
from Jiezi.Linear_algebra import operator as op

script_path = os.path.dirname(__file__)
myutils_path = os.path.join(script_path, '../../../')
sys.path.append(myutils_path)

vec = vector_numpy(2)
mat = matrix_numpy(2, 2)
sca = 2
vec.copy([1, 0])
mat.copy([[1, 1j], [1j, 1]])
# test matmulvec
assert (op.matmulvec(mat, vec.trans()).get_value() == np.array([[1], [1j]])).all(), "matmulvec is wrong"
# test vecmulmat
assert (op.vecmulmat(vec, mat).get_value() == np.array([[1, 1j]])).all(), "vecmulmat is wrong"
# test vecmulvec
assert (op.vecmulvec(vec.trans(), vec).get_value() == np.array([[1, 0], [0, 0]])).all(), "vecmulvec is wrong"
# test vecdotvec
assert op.vecdotvec(vec, vec.trans()) == 1, "vecdotvec is wrong"
# test matmulmat
assert (op.matmulmat(mat, mat).get_value() == np.array([[0, 2j], [2j, 0]])).all(), "matmulmat is wrong"
# test scamulvec
assert (op.scamulvec(sca, vec).get_value() == np.array([2, 0])).all(), "scamulvec is wrong"
# test scamulmat
assert (op.scamulmat(sca, mat).get_value() == np.array([[2, 2j], [2j, 2]])).all(), "scamulmat is wrong"
# test trimatmul
assert (op.trimatmul(mat, mat, mat).get_value() == np.array([[-2, 2j], [2j, -2]])).all(), "type nnn is wrong"
assert (op.trimatmul(mat, mat, mat, "cnn").get_value() == np.array([[2, 2j], [2j, 2]])).all(), "type one c is wrong"
assert (op.trimatmul(mat, mat, mat, "ccn").get_value() == np.array([[2, -2j], [-2j, 2]])).all(), "type two c is wrong"
assert (op.trimatmul(mat, mat, mat, "ccc").get_value() == np.array([[-2, -2j], [-2j, -2]])).all(), "type ccc is wrong"
# test addmat
assert (op.addmat(mat, mat, mat).get_value() == np.array([[3, 3j], [3j, 3]])).all(), "addmat is wrong"
# test addvec
assert (op.addvec(vec, vec, vec, vec).get_value() == np.array([4, 0])).all(), "addvec is wrong"
# test inv
mat.copy(np.array([[1, 2], [0, 1]]))
assert (op.inv(mat).get_value() == np.array([[1, -2], [0, 1]])).all(), "inv is wrong"


