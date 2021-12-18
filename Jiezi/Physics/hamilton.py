# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op


class hamilton:
    """
    this class is a storage of the whole hamiltonian matrix of CNT.
    self.__Hii is a list, Hii[i] is a matrix_numpy type object
    self.__Hi1 is a list, Hi1[i] is a matrix_numpy type object
    Hii: Hii[i]=H_{i,i}
    Hi1: Hi1[i]=H_{i-1,i}
    """
    def __init__(self, cnt):
        self.hopping_value = cnt.get_hopping_value()

        self.__H_onsite = cnt.get_hamilton_cell().tolist()
        self.__H_hopping = cnt.get_hamilton_hopping().tolist()
        self.__block_size = cnt.get_hamilton_cell().shape[0]
        self.__length = cnt.get_Trepeat()
        self.__Hii = []
        self.__Hi1 = []
        self.__Sii = []
        self.__Si1 = []

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

    def build_S(self, base_overlap=1.0):
        """
        overlap matrix of the base vector
        :param hopping_value: the non-diagonal element of H_cell
        :param base_overlap: S_AB, the overlap of base A and B
        """
        H_onsite = matrix_numpy()
        H_hopping = matrix_numpy()
        H_onsite.copy(self.__H_onsite)
        H_hopping.copy(self.__H_hopping)
        Sii = op.scamulmat(base_overlap/self.hopping_value, H_onsite)
        Si1 = op.scamulmat(base_overlap/self.hopping_value, H_hopping)
        for i in range(self.__block_size):
            Sii.set_value(i, i, 1+0j)
        for i in range(self.__length):
            self.__Sii.append(Sii)
            self.__Si1.append(Si1)
        self.__Si1.append(Si1)

    def get_Hii(self):
        return self.__Hii

    def get_Hi1(self):
        return self.__Hi1

    def get_Sii(self):
        return self.__Sii

    def get_Si1(self):
        return self.__Si1



