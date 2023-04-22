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
from Jiezi.FEM import map
# from Jiezi.NEGF.tests.fake_potential import fake_potential
# from matplotlib import pyplot as plt


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
        # coordinate is the coordinates of every atom in the whole tube. {1:[x,x,x],2:[x,x,x]...}
        # whole_length is the real length of the whole cnt
        self.__nn = cnt.get_nn()
        self.__layertolayer = cnt.get_layertolayer()
        self.__total_neighbor = cnt.get_total_neighbor()
        self.__length = cnt.get_Trepeat()
        self.__coordinate = cnt.get_coordinate()
        self.__whole_length = cnt.get_length()
        self.__onsite = onsite
        self.__hopping = hopping
        self.__Hii = []
        self.__Hi1 = []
        self.__Sii = []
        self.__Si1 = []
        self.__H00_L = matrix_numpy()
        self.__H00_R = matrix_numpy()
        self.__S00 = matrix_numpy()

    def build_single_H_cell(self, layer_number):
        """
        cell hamilton matrix is the matrix of a single cell, which is the diagonal element of the big matrix H
        the electrostatic potential phi has not been added to this matrix in this function
        :return: matrix_numpy type, hamilton matrix of one single cell
        """
        # put the hopping and onsite value in the cell hamilton array
        # the size of the cell hamilton array is nn * nn
        cell_hamilton = np.zeros((self.__nn, self.__nn))
        # onsite value is the diagonal element
        for i in range(self.__nn):
            # # the following lines is for test
            # atom_number = i + layer_number * self.__nn + 1
            # fake_phi = fake_potential(self.__coordinate[atom_number][2], self.__whole_length)
            # cell_hamilton[i, i] = self.__onsite + fake_phi
            cell_hamilton[i, i] = self.__onsite

        for row, value in self.__total_neighbor.items():
            for column in value:
                cell_hamilton[row - 1, column - 1] = self.__hopping
                cell_hamilton[column - 1, row - 1] = self.__hopping
        # TODO: think create a matrix with base class instead of numpy matrix
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
            # this hopping matrix is the right-top one, the left-bottom one is its dagger
            hopping_hamilton[element[0] - 1, element[1] - self.__nn - 1] = self.__hopping
        single_H_hopping = matrix_numpy()
        single_H_hopping.copy(hopping_hamilton)
        return single_H_hopping

    def build_H(self):
        """
        H is self.__length*self.__length
        H[i, i] = hamilton_onsite, the size of which is self.__block_size*self.__block_size
        H[i, i+1] = hamilton_hopping, the size of which is self.__block_size*self.__block_size
        H[i+1, i] = dagger(hamilton_hopping), the size of which is self.__block_size*self.__block_size
        """
        for layer_number in range(self.__length):
            H_hopping = self.build_single_H_hopping()
            H_temp = self.build_single_H_cell(layer_number)
            self.__Hii.append(H_temp)
            self.__Hi1.append(H_hopping)
        self.__Hi1.append(self.build_single_H_hopping())

    def build_S(self, base_overlap=0.018):
        """
        overlap matrix of the base vector
        :param base_overlap: S_AB, the overlap of base A and B
        """
        for i in range(self.__length):
            H_onsite = self.build_single_H_cell(0)
            H_hopping = self.build_single_H_hopping()
            Sii = op.scamulmat(base_overlap / self.__hopping, H_onsite)
            for i in range(self.__nn):
                Sii.set_value(i, i, 1.0 + 0.0j)
            Si1 = op.scamulmat(base_overlap / self.__hopping, H_hopping)
            self.__Sii.append(Sii)
            self.__Si1.append(Si1)
        self.__Si1.append(op.scamulmat(base_overlap / self.__hopping, self.build_single_H_hopping()))
        self.__S00 = self.__Sii[0]

    def H_add_phi(self, dict_cell, u_cell, cell_co, num_radius, num_z, r_oxide, z_total):
        """
        add electrostatic potential to the elements on the main diagonal line of the Hii according to the
        atom position;
        the function map.projection is used to get the potential based on the atom position.
        :param dict_cell: refer to function map.projection
        :param u_cell: refer to function map.projection
        :param cell_co: refer to function map.projection
        :param num_radius: refer to function map.projection
        :param num_z: refer to function map.projection
        :param r_oxide: refer to function map.projection
        :param z_total: refer to function map.projection
        :return: layer_phi_list. the average value of phi added on each layer.
        layer_phi_list[layer_index] is average phi added on this layer
        the onsite value of every atom is changed by the potential phi
        """
        layer_phi_list = [float] * self.__length
        for layer_number in range(self.__length):
            layer_phi = 0.0
            for i in range(self.__nn):
                atom_number = i + layer_number * self.__nn + 1
                # atom_coord is a list [x, y, z]
                atom_coord = self.__coordinate[atom_number]
                phi_atom_i = map.projection(dict_cell, u_cell, atom_coord, cell_co, num_radius, num_z, r_oxide, z_total)
                value = self.__Hii[layer_number].get_value(i, i) - phi_atom_i
                self.__Hii[layer_number].set_value(i, i, value)
                layer_phi += phi_atom_i
            layer_phi_list[layer_number] = layer_phi / self.__nn
        return layer_phi_list

    def H_defect_band(self, cell_start, cell_repeat):
        # link several cell together to form a supercell in order to satisfy a reasonable defect density
        atom_amount = self.__nn * cell_repeat
        Hii_defect = matrix_numpy(atom_amount, atom_amount)
        Hi1_defect = matrix_numpy(atom_amount, atom_amount)
        Sii_defect = matrix_numpy(atom_amount, atom_amount)
        Si1_defect = matrix_numpy(atom_amount, atom_amount)

        for i in range(cell_repeat):
            Hii_defect.set_block_value(i * self.__nn, (i + 1) * self.__nn,
                                       i * self.__nn, (i + 1) * self.__nn,
                                       self.__Hii[i + cell_start])
            Sii_defect.set_block_value(i * self.__nn, (i + 1) * self.__nn,
                                       i * self.__nn, (i + 1) * self.__nn,
                                       self.__Sii[i + cell_start])
            if i < cell_repeat - 1:
                Hii_defect.set_block_value(i * self.__nn, (i + 1) * self.__nn,
                                           (i + 1) * self.__nn, (i + 2) * self.__nn,
                                           self.__Hi1[i + 1 + cell_start])
                Hii_defect.set_block_value((i + 1) * self.__nn, (i + 2) * self.__nn,
                                           i * self.__nn, (i + 1) * self.__nn,
                                           self.__Hi1[i + 1 + cell_start].dagger())

                Sii_defect.set_block_value(i * self.__nn, (i + 1) * self.__nn,
                                           (i + 1) * self.__nn, (i + 2) * self.__nn,
                                           self.__Si1[i + 1 + cell_start])
                Sii_defect.set_block_value((i + 1) * self.__nn, (i + 2) * self.__nn,
                                           i * self.__nn, (i + 1) * self.__nn,
                                           self.__Si1[i + 1 + cell_start].dagger())
        Hi1_defect.set_block_value((cell_repeat - 1) * self.__nn, cell_repeat * self.__nn,
                                   0, self.__nn,
                                   self.__Hi1[0])
        Si1_defect.set_block_value((cell_repeat - 1) * self.__nn, cell_repeat * self.__nn,
                                   0, self.__nn,
                                   self.__Si1[0])
        return Hii_defect, Hi1_defect, Sii_defect, Si1_defect

    def H_add_defect(self, defect_index, defect_energy):
        for index in defect_index:
            layer_index = (index - 1) // self.__nn
            index_inter = index - layer_index * self.__nn
            # change the onsite value of Hii
            self.__Hii[layer_index].set_value(index_inter - 1, index_inter - 1, defect_energy)
            # change the hopping value of Hii
            for index_neighbor_inter in self.__total_neighbor[index_inter]:
                self.__Hii[layer_index].set_value(index_inter - 1, index_neighbor_inter - 1, 0)
                self.__Hii[layer_index].set_value(index_neighbor_inter - 1, index_inter - 1, 0)
            # change value of Hi1 if the defect atom is between two layers
            for link_pair in self.__layertolayer:
                m, n = link_pair
                if index_inter == m:
                    self.__Hi1[layer_index + 1].set_value(index_inter - 1, n - self.__nn - 1, 0)
                if index_inter + self.__nn == n:
                    self.__Hi1[layer_index].set_value(m - 1, index_inter - 1, 0)

    def H_readw90(self, M0_w90, M1_w90):
        # size_M is the size of the big matrix contains CNT+graphene
        size_M = M0_w90.get_size()[0]
        # the initialized value of S00 is self.__Sii[0], but for this case, S00 need to be identical matrix
        self.__S00.copy(np.eye(size_M))
        # set the block value of Hii and Hi1 in channel scope
        Hii_channel = M0_w90.get_value(40, 72, 40, 72)
        Hi1_channel = M1_w90.get_value(40, 72, 40, 72)
        for zz in range(self.__length):
            self.__Hii[zz].copy(Hii_channel)
        for zz in range(1, self.__length):
            self.__Hi1[zz].copy(Hi1_channel)
        # set the coupling matrix between channel and left contact
        self.__Hi1[0] = M1_w90.get_value(0, 72, 40, 72)
        # set the coupling matrix between channel and right contact
        self.__Hi1[self.__length] = M1_w90.get_value(40, 72, 0, 72)
        # set H00 used for surface_gf computation
        self.__H00_L.copy(M0_w90.get_value())
        self.__H00_R.copy(M0_w90.get_value())


    def get_Hii(self):
        return self.__Hii

    def get_Hi1(self):
        return self.__Hi1

    def get_Sii(self):
        return self.__Sii

    def get_Si1(self):
        return self.__Si1
