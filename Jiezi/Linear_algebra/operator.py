# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
from Jiezi.Linear_algebra.vector import vector
from Jiezi.Linear_algebra.matrix import matrix


class operator(matrix, vector):
    def __init__(self):
        return

    def matmulvec(self, mat: matrix, vec: vector):
        """
        mat(m, n) * vec(n, 1) = vec(m, 1)
        vec is column vector
        """
        assert mat.get_size()[0] == vec.get_size()[0], "The size of matrix and vector are not compatible!"
        temp = vector(vec.get_size()[0])
        temp.copy(np.dot(mat.get_value(), vec.get_value()))
        return temp

    def vecmulmat(self, vec: vector, mat: matrix):
        """
        vec(1, m) * mat(m, n) = vec(1, n)
        vec is row vector
        """
        assert vec.get_size()[0] == mat.get_size()[0], "The size of matrix and vector are not compatible!"
        temp = vector(mat.get_size()[1])
        temp.copy(np.dot(vec.get_value(), mat.get_value()))
        return temp

    def vecmulvec(self, vec1: vector, vec2: vector):
        """
        vec1(m, 1) mul vec2(1, n) = mat(m, n)
        vec1 is column vector
        vec2 is row vector
        """
        assert vec1.get_size()[1] == 1, "The first vector must be column vector!"
        assert len(vec2.get_size()) == 1, "The second vector must be row vector!"
        temp = matrix(vec1.get_size()[0], vec2.get_size()[0])
        temp.copy(np.multiply(vec1.get_value(), vec2.get_value()))
        return temp

    def vecdotvec(self, vec1: vector, vec2: vector):
        """
        vec1(1, m) * vec2(m, 1) = scalar
        """
        assert len(vec1.get_size()) == 1, "The first vector must be row vector!"
        assert vec2.get_size()[1] == 1, "The second vector must be column vector!"
        return np.dot(vec1.get_value(), vec2.get_value())

    def matmulmat(self, mat1: matrix, mat2: matrix):
        """
        mat1(m, n) * mat2(n, k) = mat(m, k)
        """
        assert mat1.get_size()[1] == mat2.get_size()[0], "The size of matrix are not compatible!"
        temp = matrix(mat1.get_size()[0], mat2.get_size()[1])
        temp.copy(np.matmul(mat1.get_value(), mat2.get_value()))
        return temp

    def scamulvec(self, scalar: complex, vec: vector):
        """
        scalar * vec
        """
        temp = vector(vec.get_size()[0])
        temp.copy(np.multiply(scalar, vec.get_value()))
        return temp

    def scamulmat(self, scalar: complex, mat: matrix):
        """
        scalar * mat
        """
        temp = matrix(mat.get_size()[0], mat.get_size()[1])
        temp.copy(np.multiply(scalar, mat.get_value()))
        return temp

    def trimatmul(self, mat1: matrix, mat2: matrix, mat3: matrix, type="nnn"):
        if type[0] == "c":
            mat1 = mat1.conjugate()
        if type[1] == "c":
            mat2 = mat2.conjugate()
        if type[2] == "c":
            mat3 = mat3.conjugate()
        return self.matmulmat(self.matmulmat(mat1, mat2), mat3)

    def addmat(self, *mat: matrix):
        temp = matrix(mat[0].get_size()[0], mat[0].get_size()[1])
        for mat_i in mat:
            temp.copy(np.add(temp.get_value(), mat_i.get_value()))
        return temp

    def addvec(self, *vec: vector):
        temp = vector(vec[0].get_size()[0])
        for vec_i in vec:
            temp.copy(np.add(temp.get_value(), vec_i.get_value()))
        return temp

    def eigenvalue(self, mat: matrix):
        return np.linalg.eig(mat.get_value())[0]

    def eigenvec(self, mat: matrix):
        temp = matrix(mat.get_size()[0], mat.get_size()[1])
        temp.copy(np.linalg.eig(mat.get_value())[1])
        return temp
