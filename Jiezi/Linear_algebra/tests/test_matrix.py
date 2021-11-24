# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import sys
import os

script_path = os.path.dirname(__file__)
myutils_path = os.path.join(script_path, '../../../')
sys.path.append(myutils_path)

import numpy as np
from Jiezi.Linear_algebra.matrix_numpy import matrix_numpy

# test init
mat = matrix_numpy(2, 3)
assert (mat.get_value() == np.array([[0, 0, 0], [0, 0, 0]])).all(), "init is wrong"
# test get_size
assert mat.get_size() == (2, 3), "get_size is wrong"
# test set_value
for m in range(2):
    for n in range(3):
        mat.set_value(m, n, complex(m, n + 1))
assert (mat.get_value() == np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), "set_value is wrong"
# test get_value
assert mat.get_value(1, 2) == 1 + 3j, "get_value(arg1, arg2) is wrong"
assert (mat.get_value(0, 1, 1, 3) == np.array([[0 + 2j, 0 + 3j]])).all(), "get_value(arg1, arg2, arg3, arg4) is wrong"
# test imaginary
assert (mat.imaginary().get_value() == np.array([[1, 2, 3], [1, 2, 3]])).all(), "imaginary is wrong"
assert (mat.get_value() == np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
    "original value can not be changed"
# test real
assert (mat.real().get_value() == np.array([[0, 0, 0], [1, 1, 1]])).all(), "real is wrong"
assert (mat.get_value() == np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
    "original value can not be changed"
# test transpose
assert (mat.trans().get_value() == np.array([[0 + 1j, 1 + 1j], [0 + 2j, 1 + 2j], [0 + 3j, 1 + 3j]])).all(), \
    "transpose is wrong"
assert (mat.get_value() == np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
    "original value can not be changed"
# test conjugate
assert (mat.conjugate().get_value() == np.array([[0 - 1j, 0 - 2j, 0 - 3j], [1 - 1j, 1 - 2j, 1 - 3j]])).all(), \
    "conjugate is wrong"
assert (mat.get_value() == np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
    "original value can not be changed"
# test dagger
assert (mat.dagger().get_value() == np.array([[0 - 1j, 1 - 1j], [0 - 2j, 1 - 2j], [0 - 3j, 1 - 3j]])).all(), \
    "dagger is wrong"
assert (mat.get_value() == np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
    "original value can not be changed"
# test negative
assert (mat.nega().get_value() == np.array([[0 - 1j, 0 - 2j, 0 - 3j], [-1 - 1j, -1 - 2j, -1 - 3j]])).all(), \
    "negative is wrong"
assert (mat.get_value() == np.array([[0 + 1j, 0 + 2j, 0 + 3j], [1 + 1j, 1 + 2j, 1 + 3j]])).all(), \
    "original value can not be changed"
# test identity
ele = matrix_numpy(3, 3)
ele.identity()
assert (ele.get_value() == np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).all(), "identity is wrong"
# test trace
assert ele.tre() == 3, "trace is wrong"
# test det
assert ele.det() == 1, "det is wrong"
# test copy
src = np.array([[1. + 2.j, 2. + 3.j, 2]])
mat.copy(src)
assert (mat.get_value() == np.array([[1. + 2.j, 2. + 3.j, 2]])).all(), "copy is wrong"
# test eigenvalue and eigenvec
a = np.diag((3, 2, 1))
mat.copy(a)
assert (mat.eigenvalue().get_value() == [1, 2, 3]).all(), "eigenvalue is wrong"
assert (mat.eigenvec().get_value() == np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])).all(), "eigenvec is wrong"
