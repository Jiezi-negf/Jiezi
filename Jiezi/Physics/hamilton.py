# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

from Jiezi.Linear_algebra.vector_numpy import vector_numpy
from Jiezi.Linear_algebra.matrix_numpy import matrix_numpy
from Jiezi.Linear_algebra import operator as op


class hamilton:
    """
    this class is a storage of the whole hamiltonian matrix of CNT.
    self.__Hii is a list, Hii[i] is a matrix_numpy type object
    self.__Hi1 is a list, Hi1[i] is a matrix_numpy type object
    Hii: Hii[i]=H_{i,i}
    Hi1: Hi1[i]=H_{i-1,i}
    """
    def __init__(self, hamilton_cell, hamilton_hopping, size, length):
        self.__H_onsite = hamilton_cell.tolist()
        self.__H_hopping = hamilton_hopping.tolist()
        self.__block_size = size
        self.__length = length
        self.__Hii = []
        self.__Hi1 = []
        self.__Sii = []
        return

    def build_H(self):
        """
        H is self.__length*self.__length
        H[i, i] = hamilton_onsite, the size of which is self.__block_size*self.__block_size
        H[i, i+1] = hamilton_hopping, the size of which is self.__block_size*self.__block_size
        H[i+1, i] = dagger(hamilton_hopping), the size of which is self.__block_size*self.__block_size
        """
        H_onsite = matrix_numpy()
        H_onsite.copy(self.__H_onsite)
        H_hopping = matrix_numpy()
        H_hopping.copy(self.__H_hopping)
        for i in range(self.__length):
            self.__Hii.append(H_onsite)
            self.__Hi1.append(H_hopping)
        self.__Hi1.append(H_hopping)

    def build_S(self, hopping_value, base_overlap=1.0):
        """
        overlap matrix of the base vector
        :param hopping_value: the non-diagonal element of H_cell
        :param base_overlap: S_AB, the overlap of base A and B
        """
        H_onsite = matrix_numpy()
        H_onsite.copy(self.__H_onsite)
        S = op.scamulmat(base_overlap/hopping_value, H_onsite)
        for i in range(self.__block_size):
            S.set_value(i, i, 1+0j)
        for i in range(self.__length):
            self.__Sii.append(S)

    def get_hamilton_onsite(self):
        return self.__Hii

    def get_hamilton_hopping(self):
        return self.__Hi1

    def get_S(self):
        return self.__Sii



