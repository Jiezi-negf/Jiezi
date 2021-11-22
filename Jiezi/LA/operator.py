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
        temp = vector_numpy(vec.get_size()[0])
        temp.copy(np.dot(mat.get_value(), vec.get_value()))
        return temp


def vecmulmat(vec: vector, mat: matrix):
    """
    vec(1, m) * mat(m, n) = vec(1, n)
    vec is row vector
    """
    if isinstance(vec, vector_numpy) and isinstance(mat, matrix_numpy):
        assert vec.get_size()[0] == mat.get_size()[0], "The size of matrix and vector are not compatible!"
        temp = vector_numpy(mat.get_size()[1])
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
        assert len(vec2.get_size()) == 1, "The second vector must be row vector!"
        temp = matrix_numpy(vec1.get_size()[0], vec2.get_size()[0])
        temp.copy(np.multiply(vec1.get_value(), vec2.get_value()))
        return temp


def vecdotvec(vec1: vector, vec2: vector):
    """
    vec1(1, m) * vec2(m, 1) = scalar
    """
    if isinstance(vec1, vector_numpy) and isinstance(vec2, vector_numpy):
        assert len(vec1.get_size()) == 1, "The first vector must be row vector!"
        assert vec2.get_size()[1] == 1, "The second vector must be column vector!"
        return np.dot(vec1.get_value(), vec2.get_value())


def matmulmat(mat1: matrix, mat2: matrix):
    """
    mat1(m, n) * mat2(n, k) = mat(m, k)
    """
    if isinstance(mat1, matrix_numpy) and isinstance(mat2, matrix_numpy):
        assert mat1.get_size()[1] == mat2.get_size()[0], "The size of matrix are not compatible!"
        temp = matrix_numpy(mat1.get_size()[0], mat2.get_size()[1])
        temp.copy(np.matmul(mat1.get_value(), mat2.get_value()))
        return temp


def scamulvec(scalar: complex, vec: vector):
    """
    scalar * vec
    """
    if isinstance(vec, vector_numpy):
        temp = vector_numpy(vec.get_size()[0])
        temp.copy(np.multiply(scalar, vec.get_value()))
        return temp


def scamulmat(scalar: complex, mat: matrix):
    """
    scalar * mat
    """
    if isinstance(mat, matrix_numpy):
        temp = matrix_numpy(mat.get_size()[0], mat.get_size()[1])
        temp.copy(np.multiply(scalar, mat.get_value()))
        return temp


def trimatmul(mat1: matrix, mat2: matrix, mat3: matrix, type="nnn"):
    if isinstance(mat1, matrix_numpy) and isinstance(mat2, matrix_numpy) and isinstance(mat3, matrix_numpy):
        if type[0] == "c":
            mat1 = mat1.conjugate()
        if type[1] == "c":
            mat2 = mat2.conjugate()
        if type[2] == "c":
            mat3 = mat3.conjugate()
        return matmulmat(matmulmat(mat1, mat2), mat3)


def addmat(*mat: matrix):
    if isinstance(mat[0], matrix_numpy):
        temp = matrix_numpy(mat[0].get_size()[0], mat[0].get_size()[1])
        for mat_i in mat:
            temp.copy(np.add(temp.get_value(), mat_i.get_value()))
        return temp


def addvec(*vec: vector):
    if isinstance(vec[0], vector_numpy):
        temp = vector_numpy(vec[0].get_size()[0])
        for vec_i in vec:
            temp.copy(np.add(temp.get_value(), vec_i.get_value()))
        return temp


def inv(mat):
    if isinstance(mat, matrix_numpy):
        temp = matrix_numpy(mat.get_size()[0], mat.get_size()[1])
        temp.copy(np.linalg.inv(mat.get_value()))
        return temp
