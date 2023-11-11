# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import sys
import numpy as np
sys.path.append("../../../")
from Jiezi.FEM.shape_function import shape_function

coord = [1, 2, 1]
dof_coor = [[4, 5, 6], [2, 7, 1], [5, 2, 3], [1, 3, 7]]
co_shapefunc = shape_function(dof_coor)
matrix_1 = np.arange(16).reshape(4, 4)
for i in range(4):
    matrix_1[i, :] = [1] + dof_coor[i]
matrix_2 = np.array(co_shapefunc)
print(matrix_1)
print(co_shapefunc)
print(matrix_2)
print(np.dot(matrix_1, matrix_2))
