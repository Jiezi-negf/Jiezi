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
from Jiezi.Graph.builder import CNT
import numpy as np


class hamilton:
    """
    this class is a storage of the whole hamiltonian matrix of CNT.
    self.__Hii is a list, Hii[i] is a matrix_numpy type object
    self.__Hi1 is a list, Hi1[i] is a matrix_numpy type object
    Hii: Hii[i]=H_{i,i}
    Hi1: Hi1[i]=H_{i-1,i}
    """
    def __init__(self, cnt: CNT, onsite=-0.28, hopping=-2.97):
        # size of both cell hamilton matrix and hopping hamilton matrix are nn * nn
        # layertolayer is to build hopping hamilton
        # total_neighbor is to build cell hamilton
        # length is the number of layers/cells
        self.__nn = cnt.get_nn()
        self.__layertolayer = cnt.get_layertolayer()
        self.__total_neighbor = cnt.get_total_neighbor()
        self.__length = cnt.get_Trepeat()
        self.__onsite = onsite
        self.__hopping = hopping
        self.__Hii = []
        self.__Hi1 = []
        self.__Sii = []
        self.__Si1 = []

    def build_single_H_cell(self, phi):
        """
        cell hamilton matrix is the matrix of a single cell, which is the diagonal element of the big matrix H
        :param phi: the electronic potential, which will influence the diagonal element of the single cell matrix
        it should be noticed that here I just regard all atoms of a single cell as a block, which have no difference
        with each other. So the potential added on the matrix element is the same one.
        But the potential added on different layer will be different.
        In other word, in this version, the potential varies only along the translation vector (t_1, t_2)
        :return: matrix_numpy type, hamilton matrix of one single cell
        """
        # put the hopping and onsite value in the cell hamilton array
        # the size of the cell hamilton array is nn * nn
        cell_hamilton = np.zeros((self.__nn, self.__nn))
        # onsite value is the diagonal element
        for i in range(self.__nn):
            cell_hamilton[i, i] = self.__onsite + phi
        for row, value in self.__total_neighbor.items():
            for column in value:
                cell_hamilton[row - 1, column - 1] = self.__hopping
                cell_hamilton[column - 1, row - 1] = self.__hopping
        single_H_cell = matrix_numpy()
        single_H_cell.copy(cell_hamilton)
        return single_H_cell

    def build_single_H_hopping(self):
        """
        build the hopping hamilton matrix
        :return: matrix_numpy type, hopping hamilton matrix
        """
        # construct hopping hamilton matrix
        hopping_hamilton = np.zeros((self.__nn, self.__nn))
        # layertolayer list follows the rule that initial(small) layer number first, next(big) layer number then
        for element in self.__layertolayer:
            # this hopping matrix is the right-top one, the left-bottom one is its conjugate
            hopping_hamilton[element[0] - 1, element[1] - self.__nn - 1] = self.__hopping
        single_H_hopping = matrix_numpy()
        single_H_hopping.copy(hopping_hamilton)
        return single_H_hopping

    def build_H(self, phi_list):
        """
        H is self.__length*self.__length
        H[i, i] = hamilton_onsite, the size of which is self.__block_size*self.__block_size
        H[i, i+1] = hamilton_hopping, the size of which is self.__block_size*self.__block_size
        H[i+1, i] = dagger(hamilton_hopping), the size of which is self.__block_size*self.__block_size
        :param phi_list: list, electrical potential set, phi_list[i] is the potential added on the ith layer
        """
        H_hopping = self.build_single_H_hopping()
        for i in range(self.__length):
            self.__Hii.append(self.build_single_H_cell(phi_list[i]))
            self.__Hi1.append(H_hopping)
        self.__Hi1.append(H_hopping)

    def build_S(self, base_overlap=0.018):
        """
        overlap matrix of the base vector
        :param base_overlap: S_AB, the overlap of base A and B
        """
        H_onsite = self.build_single_H_cell(phi=0)
        H_hopping = self.build_single_H_hopping()
        Sii = op.scamulmat(base_overlap / self.__hopping, H_onsite)
        Si1 = op.scamulmat(base_overlap / self.__hopping, H_hopping)
        for i in range(self.__nn):
            Sii.set_value(i, i, 1 + 0j)
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
