# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import numpy as np
from Jiezi.Linear_algebra.base_linalg import matrix


class matrix_numpy(matrix):
    def __init__(self, row: int = 2, column: int = 2):
        self.__value = np.zeros((row, column), dtype=complex, order="C")
        self.__row = row
        self.__column = column

    def get_size(self):
        return self.__value.shape

    def set_value(self, row, column, value):
        self.__value[row, column] = value

    def get_value(self, *index):
        """
        return the matrix as numpy array
        if index is None, then the numpy type result will be returned
        the reason why I return np.copy rather than return the self.__value is:
        >>mat1 = matrix(2, 2)
        >>mat2 = matrix(2, 2)
        >>mat1.copy([[1, 1j], [1j, 1]])
        >>mat2.copy(mat1.get_value())
        if I change the element of mat2, the mat1's element will not be influenced,
        but if I return the self.__value directly, it will be changed with mat1
        """
        if index == ():
            return np.copy(self.__value)
        elif len(index) == 2:
            return np.copy(self.__value[index[0], index[1]])
        else:
            return np.copy(self.__value[index[0]:index[1], index[2]:index[3]])

    def imaginary(self):
        temp = matrix_numpy(self.__row, self.__column)
        temp.copy(self.get_value().imag)
        return temp

    def real(self):
        temp = matrix_numpy(self.__row, self.__column)
        temp.copy(self.get_value().real)
        return temp

    def trans(self):
        temp = matrix_numpy(self.__row, self.__column)
        temp.copy(np.transpose(self.get_value()))
        return temp

    def conjugate(self):
        temp = matrix_numpy(self.__row, self.__column)
        temp.copy(np.conjugate(self.get_value()))
        return temp

    def dagger(self):
        temp = matrix_numpy(self.__row, self.__column)
        temp.copy(np.conjugate(self.get_value()))
        temp.copy(np.transpose(temp.__value))
        return temp

    def nega(self):
        temp = matrix_numpy(self.__row, self.__column)
        temp.copy(np.negative(self.get_value()))
        return temp

    def identity(self):
        assert self.__row == self.__column, "identity matrix must be square"
        self.__value = np.eye(self.__row, self.__column)

    def tre(self):
        """
        trace
        """
        return np.trace(self.get_value())

    def det(self):
        """
        determination
        """
        return np.linalg.det(self.get_value())

    def copy(self, source):
        """
        source must be the numpy type parameter rather than matrix or vector
        """
        self.__value = np.asarray(source, dtype=complex)

    def print(self):
        print(self.__value)
