# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys

sys.path.append("../../../")

from Jiezi.Graph import builder
from Jiezi.Physics import hamilton
from Jiezi.Physics.band import subband
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.rgf import rgf
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.Physics.common import *
import numpy as np


# construct the structure
cnt = builder.CNT(n=4, m=0, Trepeat=3, nonideal=False)
cnt.construct()
radius_tube = cnt.get_radius()
length_single_cell = cnt.get_singlecell_length()
volume_cell = math.pi * radius_tube ** 2 * length_single_cell

# build hamilton matrix
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)

# compute the eigenvalue(subband energy) and eigenvector(transformation matrix)
# E_subband is a list, the length of which is equal to number of layers.
# The length of E_subband[i] is equal to atoms in every single layer.
# U is a list, the length of which is equal to number of layers.
# The size of U[i] is equal to every single block matrix of H
E_subband, U = subband(H, k=0)
# print(E_subband[0].get_value())
print(U[0].get_value())

# compute the mode space basis to decrease the size of H
nm = E_subband[0].get_size()[0]
Hii_new, Hi1_new, Sii_new, form_factor = mode_space(H, U, nm-6)

# print(U_uni[0].get_value())
# print(U_new[0].get_value())
#
# U_if = op.matmulmat(U[0].dagger(), U[0]).get_value()
# U_uni_if = op.matmulmat(U_uni[0].dagger(), U_uni[0]).get_value()
# U_uni_if_2 = op.matmulmat(U_uni[0], U_uni[0].dagger()).get_value()
# U_new_if = op.matmulmat(U_new[0].dagger(), U_new[0]).get_value()
# U_new_if_2 = op.matmulmat(U_new[0], U_new[0].dagger()).get_value()
a = np.array([[1, 0, 1],[1, 1, 0],[0, 1, 1]])
b = np.array([[1, 0, 0, 1],[1, 1, 0, 0],[0, 1, 1, 0], [0, 0, 1, 1]])
c = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 3]])
test_a = np.dot(np.linalg.eig(a)[1].conjugate().T, np.linalg.eig(a)[1])
test_b = np.dot(np.linalg.eig(b)[1].conjugate().T, np.linalg.eig(b)[1])
test_c = np.dot(np.linalg.eig(a)[1].conjugate().T, c, np.linalg.eig(a)[1])
print(test_a)
print(test_b)

