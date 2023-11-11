# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

# As we all know if U(size of which is n*n) is a unitary matrix, then every column of U is orthogonal with each other,
# i.e. U^\dagger*U=I, and every column of U is also orthogonal.
# Even if we reduce some column from U(the size of reduced U becomes n*m), U^\dagger*U=I(m*m) will not be changed.
# the question is, after this reducing process,
# will the rows of this matrix still keep orthogonal i.e. U*U\dagger=I(n*n)?

from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op
import numpy as np

# a is a hermite matrix
a = np.array([[1, 7, 3, 4],
              [7, 2, 5, 6],
              [3, 5, 5, 7],
              [4, 6, 7, 1]])
# u is constituted by the eigenvector of a
# every vector of u is the eigenvector of a
# as a is hermite, u is unitary
eigenValue, u = np.linalg.eig(a)
print(eigenValue, u)
# test if U is unitary
U = matrix_numpy()
U.copy(u)
U_dagger = U.dagger()
print("U*U\dagger:", '\n', op.matmulmat(U_dagger, U).get_value())
print("U\dagger*U:", '\n', op.matmulmat(U, U_dagger).get_value())
# test if the rows of reduced U is still orthogonal
U_reduced = matrix_numpy()
U_reduced.copy(U.get_value(0, 4, 0, 3))
print("U_reduced*U_reduced\dagger:", '\n', op.matmulmat(U_reduced, U_reduced.dagger()).get_value())
print("U_reduced\dagger*U_reduced:", '\n', op.matmulmat(U_reduced.dagger(), U_reduced).get_value())

inv_UUdagger = op.inv(op.matmulmat(U_reduced, U_reduced.dagger()))
print("inv of U_reduced*U_reduced^\dagger is:", '\n', inv_UUdagger.get_value())

# transform real space A with mode space A
Areal = matrix_numpy()
Areal.copy(a)
Amode = op.trimatmul(U_reduced, Areal, U_reduced, "cnn")
print("A in real space is:\n", Areal.get_value())
print("A in mode space is:\n", Amode.get_value())
Areal_direct = op.trimatmul(U_reduced, Amode, U_reduced, "nnc")
print("real space matrix computed back by direct inverse matrix method:", '\n', Areal_direct.get_value())

invLeft, invRight = op.general_inv(U_reduced)
rank_U_reduced = np.linalg.matrix_rank(U_reduced.get_value())
print("rank of reduced U is:", rank_U_reduced)
print("invLeft*U_reduced\dagger:", '\n', op.matmulmat(invLeft, U_reduced.dagger()).get_value())
print("U_reduced*invRight", '\n', op.matmulmat(U_reduced, invRight).get_value())
Areal_GeneralInv = op.trimatmul(invLeft, Amode, invRight, "nnn")
print("real space matrix computed back by general inverse matrix method:", '\n', Areal_GeneralInv.get_value())

mat = matrix_numpy()
mat.copy(np.array([[1, 1], [5, 4], [2, 6]]))
mat2 = op.matmulmat(mat, mat.dagger())
pinv = np.linalg.pinv(mat.dagger().get_value())
print("pinv result:", pinv)
print(np.matmul(mat.dagger().get_value(), pinv))
print(np.matmul(pinv, mat.dagger().get_value()))
print(mat2.get_value(), np.linalg.matrix_rank(mat2.get_value()), np.linalg.det(mat2.get_value()))
print(op.matmulmat(op.inv(mat2), mat2).get_value())
print(op.matmulmat(op.matmulmat(op.inv(op.matmulmat(mat, mat.dagger())), mat), mat.dagger()).get_value())
print(op.matmulmat(op.inv(op.matmulmat(mat, mat.dagger())), op.matmulmat(mat, mat.dagger())).get_value())
print(op.matmulmat(op.matmulmat(mat, mat.dagger()), op.inv(op.matmulmat(mat, mat.dagger()))).get_value())
