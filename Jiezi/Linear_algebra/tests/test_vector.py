# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys
import os
from Jiezi.Linear_algebra.vector_numpy import vector_numpy
import numpy as np

script_path = os.path.dirname(__file__)
myutils_path = os.path.join(script_path, '../../../')
sys.path.append(myutils_path)

# test init
vec = vector_numpy(3)
assert (vec.get_value() == np.array([0+0j, 0+0j, 0+0j])).all(), "init is wrong"
# test get_size
assert vec.get_size() == (3,), "get_size is wrong"
# test set_value
vec.set_value(0, 1+3j)
vec.set_value(1, 2+2j)
assert (vec.get_value() == np.array([1+3j, 2+2j, 0+0j])).all(), "set_value is wrong"
# test get_value
assert vec.get_value(2) == 0+0j, "get_value(arg1) is wrong"
assert (vec.get_value(0, 2) == np.array([1+3j, 2+2j])).all(), "get_value(arg1, arg2) is wrong"
# test imaginary
assert (vec.imaginary().get_value() == np.array([3, 2, 0])).all(), "imaginary is wrong"
assert (vec.get_value() == np.array([1+3j, 2+2j, 0+0j])).all(), "original value can not be changed"
# test real
assert (vec.real().get_value() == np.array([1, 2, 0])).all(), "real is wrong"
assert (vec.get_value() == np.array([1+3j, 2+2j, 0+0j])).all(), "original value can not be changed"
# test transpose
assert (vec.trans().get_value() == np.array([[1+3j], [2+2j], [0+0j]])).all(), "transpose is wrong"
assert (vec.get_value() == np.array([1 + 3j, 2 + 2j, 0 + 0j])).all(), "original value can not be changed"
# test conjugate
assert (vec.conjugate().get_value() == np.array([1-3j, 2-2j, 0-0j])).all(), "conjugate is wrong"
assert (vec.get_value() == np.array([1 + 3j, 2 + 2j, 0 + 0j])).all(), "original value can not be changed"
# test dagger
assert (vec.dagger().get_value() == np.array([[1-3j], [2-2j], [0-0j]])).all(), "dagger is wrong"
assert (vec.get_value() == np.array([1 + 3j, 2 + 2j, 0 + 0j])).all(), "original value can not be changed"
# test negative
assert (vec.nega().get_value() == np.array([-1-3j, -2-2j, 0])).all(), "negative is wrong"
assert (vec.get_value() == np.array([1 + 3j, 2 + 2j, 0 + 0j])).all(), "original value can not be changed"
# test copy
src = np.array([1.+2.j, 2.+3.j, 2])
vec.copy(src)
assert (vec.get_value() == np.array([1.+2.j, 2.+3.j, 2])).all(), "copy is wrong"




