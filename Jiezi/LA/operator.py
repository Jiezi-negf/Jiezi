# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
from Jiezi.LA.base_linalg import vector, matrix
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA.matrix_numpy import matrix_numpy


def matmulvec(mat: matrix, vec: vector):
    """
    mat(m, n) * vec(n, 1) = vec(m, 1)
    vec is column vector
    """
    if isinstance(mat, matrix_numpy) and isinstance(vec, vector_numpy):
        assert mat.get_size()[0] == vec.get_size()[0], "The size of matrix and vector are not compatible!"
        temp = vector_numpy(vec.get_size()[0], mat.get_type())
        temp.copy(np.dot(mat.get_value(), vec.get_value()))
        return temp


def vecmulmat(vec: vector, mat: matrix):
    """
    vec(1, m) * mat(m, n) = vec(1, n)
    vec is row vector
    """
    if isinstance(vec, vector_numpy) and isinstance(mat, matrix_numpy):
        assert vec.get_size()[1] == mat.get_size()[0], "The size of matrix and vector are not compatible!"
        temp = vector_numpy(mat.get_size()[1], mat.get_type())
        temp.copy(np.dot(vec.get_value(), mat.get_value()))
        return temp


def vecmulvec(vec1: vector, vec2: vector):
    """
    vec1(m, 1) mul vec2(1, n) = mat(m, n)
    vec1 is column vector
    vec2 is row vector
    """
    if isinstance(vec1, vector_numpy) and isinstance(vec2, vector_numpy):
        assert vec1.get_size()[1] == 1, "The first vector must be column vector!"
        assert vec2.get_size()[0] == 1, "The second vector must be row vector!"
        temp = matrix_numpy(vec1.get_size()[0], vec2.get_size()[1], vec1.get_type())
        temp.copy(np.multiply(vec1.get_value(), vec2.get_value()))
        return temp


def vecdotvec(vec1: vector, vec2: vector):
    """
    vec1(1, m) * vec2(m, 1) = scalar
    """
    if isinstance(vec1, vector_numpy) and isinstance(vec2, vector_numpy):
        assert vec1.get_size()[0] == 1, "The first vector must be row vector!"
        assert vec2.get_size()[1] == 1, "The second vector must be column vector!"
        return np.dot(vec1.get_value(), vec2.get_value())[0][0]


def matmulmat(mat1: matrix, mat2: matrix):
    """
    mat1(m, n) * mat2(n, k) = mat(m, k)
    """
    if isinstance(mat1, matrix_numpy) and isinstance(mat2, matrix_numpy):
        assert mat1.get_size()[1] == mat2.get_size()[0], ("The size of matrix are not compatible!" +
                                                          str(mat1.get_size()[1]) + ":" + str(mat2.get_size()[0]))
        temp = matrix_numpy(mat1.get_size()[0], mat2.get_size()[1], mat1.get_type())
        temp.copy(np.matmul(mat1.get_value(), mat2.get_value()))
        return temp


def matmul_sym(mat1: matrix, mat2: matrix):
    """
    the final result is a Hermite matrix, we can just calculate the diagonal element and right-up part elements
    """
    if isinstance(mat1, matrix_numpy) and isinstance(mat2, matrix_numpy):
        assert mat1.get_size()[1] == mat2.get_size()[0], "The size of matrix are not compatible!"
        temp = matrix_numpy(mat1.get_size()[0], mat2.get_size()[1], mat1.get_type())
        for i in range(mat1.get_size()[0]):
            for j in range(i, mat2.get_size()[1]):
                temp_ij = 0.0 + 0.0j
                for k in range(mat1.get_size()[1]):
                    temp_ij += mat1.get_value(i, k) * mat2.get_value(k, j)
                temp.set_value(i, j, temp_ij)
                temp_ji = complex(temp_ij.real, -temp_ij.imag)
                temp.set_value(j, i, temp_ji)
        return temp


def scamulvec(scalar, vec: vector):
    """
    scalar * vec
    """
    if isinstance(vec, vector_numpy):
        temp = vector_numpy(vec.get_size()[0], vec.get_type())
        temp.copy(np.multiply(scalar, vec.get_value()))
        return temp


def scamulmat(scalar, mat: matrix):
    """
    scalar * mat
    """
    if isinstance(mat, matrix_numpy):
        temp = matrix_numpy(mat.get_size()[0], mat.get_size()[1], mat.get_type())
        temp.copy(np.multiply(scalar, mat.get_value()))
        return temp


def trimatmul(mat1: matrix, mat2: matrix, mat3: matrix, type="nnn"):
    if isinstance(mat1, matrix_numpy) and isinstance(mat2, matrix_numpy) and isinstance(mat3, matrix_numpy):
        if type[0] == "c":
            mat1 = mat1.dagger()
        if type[1] == "c":
            mat2 = mat2.dagger()
        if type[2] == "c":
            mat3 = mat3.dagger()
        return matmulmat(matmulmat(mat1, mat2), mat3)


def addmat(*mat: matrix):
    if isinstance(mat[0], matrix_numpy):
        temp = matrix_numpy(mat[0].get_size()[0], mat[0].get_size()[1], mat[0].get_type())
        for mat_i in mat:
            temp.copy(np.add(temp.get_value(), mat_i.get_value()))
        return temp


def addvec(*vec: vector):
    if isinstance(vec[0], vector_numpy):
        temp = vector_numpy(vec[0].get_size()[0], vec[0].get_type())
        temp.copy(vec[0].get_value())
        for i in range(1, len(vec)):
            vec_i = vec[i]
            temp.copy(np.add(temp.get_value(), vec_i.get_value()))
        return temp


def inv(mat):
    if isinstance(mat, matrix_numpy):
        temp = matrix_numpy(mat.get_size()[0], mat.get_size()[1], mat.get_type())
        temp.copy(np.linalg.inv(mat.get_value()))
        return temp


def qr_decomp(mat):
    if isinstance(mat, matrix_numpy):
        temp = matrix_numpy(mat.get_size()[0], mat.get_size()[1], mat.get_type())
        temp.copy(np.linalg.qr(mat.get_value())[0])
        return temp


def general_inv(mat):
    if isinstance(mat, matrix_numpy):
        left = matmulmat(inv(matmulmat(mat, mat.dagger())), mat)
        right = matmulmat(mat.dagger(), inv(matmulmat(mat, mat.dagger())))
        return left, right
