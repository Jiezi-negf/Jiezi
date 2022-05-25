# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import numpy as np
from Jiezi.LA.base_linalg import vector


class vector_numpy(vector):
    """
    vector is defined as row vector by default
    """

    def __init__(self, n: int = 2):
        self.__value = np.zeros(n, dtype=complex, order="C").reshape(-1, 1)
        self.__size = n

    def get_size(self):
        return self.__value.shape

    def set_value(self, index, value):
        self.__value[index] = value

    def get_value(self, *index):
        """
        return the vector as numpy array
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
        elif np.size(index) == 1:
            return np.copy(self.__value[index, 0])
        else:
            return np.copy(self.__value[index[0]:index[1], 0]).reshape(-1, 1)

    def imaginary(self):
        """
        the reason why I construct the temporary variable "temp" is to avoid changing the original value,
        because the following function is just for operation, the value of "self" should not be changed
        """
        temp = vector_numpy(self.__size)
        temp.copy(self.get_value().imag)
        return temp

    def real(self):
        temp = vector_numpy(self.__size)
        temp.copy(self.get_value().real)
        return temp

    def trans(self):
        temp = vector_numpy(self.__size)
        temp.copy(self.get_value().T)
        return temp

    def conjugate(self):
        temp = vector_numpy(self.__size)
        temp.copy(np.conjugate(self.get_value()))
        return temp

    def dagger(self):
        temp = vector_numpy(self.__size)
        temp.copy(np.conjugate(self.get_value()))
        temp.copy(temp.get_value().reshape(self.__size, 1))
        return temp

    def nega(self):
        temp = vector_numpy(self.__size)
        temp.copy(np.negative(self.get_value()))
        return temp

    def copy(self, source):
        """
        source must be the numpy type parameter rather than matrix or vector
        """
        self.__value = np.asarray(source)
        self.__size = self.__value.shape

    def print(self):
        print(self.__value)
