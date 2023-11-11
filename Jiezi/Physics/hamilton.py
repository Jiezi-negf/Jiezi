# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA import operator as op
from Jiezi.Graph.builder import CNT
import numpy as np
from Jiezi.FEM import map
from scipy.optimize import minimize

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
        self.__singleCellLen = cnt.get_singlecell_length()
        self.__coordinate = cnt.get_coordinate()
        self.__whole_length = cnt.get_length()
        self.__onsite = onsite
        self.__hopping = hopping
        self.__Hii = []
        self.__Hi1 = []
        self.__Sii = []
        self.__Si1 = []
        self.__LEAD_H00_L = matrix_numpy()
        self.__LEAD_H00_R = matrix_numpy()
        self.__LEAD_H10_L = matrix_numpy()
        self.__LEAD_H10_R = matrix_numpy()
        self.__S00 = matrix_numpy()

    def build_single_H_cell(self, layer_number):
        """
        cell hamilton matrix is the matrix of a single cell, which is the diagonal element of the big matrix H
        the electrostatic potential phi has not been added to this matrix in this function
        :return: matrix_numpy type, hamilton matrix of one single cell
        """
        # put the hopping and onsite value in the cell hamilton array
        # the size of the cell hamilton array is nn * nn
        cell_hamilton = np.zeros((self.__nn, self.__nn), dtype=complex)
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
        single_H_cell = matrix_numpy()
        single_H_cell.copy(cell_hamilton)
        return single_H_cell

    def build_single_H_hopping(self):
        """
        build the hopping hamilton matrix
        :return: matrix_numpy type, hopping hamilton matrix
        """
        # construct hopping hamilton matrix
        hopping_hamilton = np.zeros((self.__nn, self.__nn), dtype=complex)
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
        self.__LEAD_H00_L.copy(self.build_single_H_cell(0).get_value())
        self.__LEAD_H00_R.copy(self.build_single_H_cell(0).get_value())
        self.__LEAD_H10_L.copy(self.build_single_H_hopping().get_value())
        self.__LEAD_H10_R.copy(self.build_single_H_hopping().get_value())

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
        self.__S00.copy(self.__Sii[0].get_value())

    def H_add_phi(self, dict_cell, u_cell, cell_co, num_radius, num_z, r_oxide, z_total, num_supercell):
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
        superLayerAmount = int(self.__length / num_supercell)
        layer_phi_list = [float] * superLayerAmount
        for layer_number in range(superLayerAmount):
            layer_phi = 0.0
            for i in range(self.__nn * num_supercell):
                atom_number = i + layer_number * self.__nn * num_supercell + 1
                # atom_coord is a list [x, y, z]
                atom_coord = self.__coordinate[atom_number]
                phi_atom_i = map.projection(dict_cell, u_cell, atom_coord, cell_co, num_radius, num_z, r_oxide, z_total)
                value = self.__Hii[layer_number].get_value(i, i) - phi_atom_i
                self.__Hii[layer_number].set_value(i, i, value)
                layer_phi += phi_atom_i
            layer_phi_list[layer_number] = layer_phi / (self.__nn * num_supercell)
        # add phi to lead matrix
        size = self.__LEAD_H00_L.get_size()[0]
        eye = matrix_numpy(size, size)
        eye.identity()
        self.__LEAD_H00_L = op.addmat(self.__LEAD_H00_L,
                                      op.scamulmat(- layer_phi_list[0], eye))
        self.__LEAD_H00_R = op.addmat(self.__LEAD_H00_R,
                                      op.scamulmat(- layer_phi_list[superLayerAmount - 1],
                                                   eye))
        return layer_phi_list

    def H_defect_band(self, cell_start, cell_repeat):
        """

        :param cell_start:
        :param cell_repeat:
        :return:
        """
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

    def H_readw90(self, Mii_total, Mi1_total, Mii_cnt, Mi1_cnt, num_supercell):
        size_M_total = Mii_total.get_size()[0]
        size_M_cnt = Mii_cnt.get_size()[0]
        size_single_total = int(size_M_total / num_supercell)
        size_single_cnt = int(size_M_cnt / num_supercell)
        superLayerAmount = int(self.__length / num_supercell)
        # redefine the length of Hii, Hi1, Sii, Si1
        self.__Hii = self.__Hii[0: superLayerAmount]
        self.__Hi1 = self.__Hi1[0: superLayerAmount + 1]
        self.__Sii = self.__Sii[0: superLayerAmount]
        self.__Si1 = self.__Si1[0: superLayerAmount + 1]
        # the initialized value of S00 is self.__Sii[0], but for this case, S00 need to be identical matrix
        self.__S00.copy(np.eye(size_M_total))
        # set the block value of Hii, Hi1, Sii, Si1 in channel scope
        for zz in range(superLayerAmount):
            self.__Hii[zz].copy(Mii_cnt.get_value())
            self.__Sii[zz].copy(np.eye(size_M_cnt))
        for zz in range(1, superLayerAmount):
            self.__Hi1[zz].copy(Mi1_cnt.get_value())
        for zz in range(0, superLayerAmount + 1):
            self.__Si1[zz].copy(np.zeros([size_M_cnt, size_M_cnt]))
        # # set the coupling matrix between channel and left contact
        # self.__Hi1[0] = matrix_numpy(144, 64)
        # self.__Hi1[0].set_block_value(0, 144, 0, 32, Mi1_total.get_value(0, 144, 40, 72))
        # self.__Hi1[0].set_block_value(0, 144, 32, 64, Mi1_total.get_value(0, 144, 112, 144))
        # # set the coupling matrix between channel and right contact
        # self.__Hi1[superLayerAmount] = matrix_numpy(64, 144)
        # self.__Hi1[superLayerAmount].set_block_value(0, 32, 0, 144, Mi1_total.get_value(40, 72, 0, 144))
        # self.__Hi1[superLayerAmount].set_block_value(32, 64, 0, 144, Mi1_total.get_value(112, 144, 0, 144))

        # set the coupling matrix between channel and left contact - general way
        self.__Hi1[0] = matrix_numpy(size_M_total, size_M_cnt)
        for i in range(num_supercell):
            self.__Hi1[0].set_block_value(0, size_M_total, size_single_cnt * i, size_single_cnt * (i + 1),
                                          Mi1_total.get_value(0, size_M_total,
                                                              size_single_total * (i + 1) - size_single_cnt,
                                                              size_single_total * (i + 1)))
        # set the coupling matrix between channel and right contact - general way
        self.__Hi1[superLayerAmount] = matrix_numpy(size_M_cnt, size_M_total)
        for i in range(num_supercell):
            self.__Hi1[superLayerAmount].set_block_value(size_single_cnt * i, size_single_cnt * (i + 1),
                                                         0, size_M_total,
                                                         Mi1_total.get_value(
                                                             size_single_total * (i + 1) - size_single_cnt,
                                                             size_single_total * (i + 1), 0, size_M_total))

        # set H00 and H10 used for surface_gf computation
        self.__LEAD_H00_L.copy(Mii_total.get_value())
        self.__LEAD_H00_R.copy(Mii_total.get_value())
        self.__LEAD_H10_L.copy(Mi1_total.get_value())
        self.__LEAD_H10_R.copy(Mi1_total.get_value())

    def get_Hii(self):
        return self.__Hii

    def get_Hi1(self):
        return self.__Hi1

    def get_Sii(self):
        return self.__Sii

    def get_Si1(self):
        return self.__Si1

    def get_S00(self):
        return self.__S00

    def get_lead_H00(self):
        return self.__LEAD_H00_L, self.__LEAD_H00_R

    def get_lead_H10(self):
        return self.__LEAD_H10_L, self.__LEAD_H10_R


def H_extendSize(Hii_cell, Hi1_cell, factor):
    size_old = Hii_cell.get_size()[0]
    size_new = int(size_old * factor)
    Hii_new = matrix_numpy(size_new, size_new, "float")
    Hi1_new = matrix_numpy(size_new, size_new, "float")
    for i in range(factor):
        Hii_new.set_block_value(i * size_old, (i + 1) * size_old, i * size_old, (i + 1) * size_old,
                                Hii_cell.get_value())
        if i > 0:
            Hii_new.set_block_value((i - 1) * size_old, i * size_old, i * size_old, (i + 1) * size_old,
                                    Hi1_cell.get_value())
            Hii_new.set_block_value(i * size_old, (i + 1) * size_old, (i - 1) * size_old, i * size_old,
                                    Hi1_cell.trans().get_value())
    Hi1_new.set_block_value((factor - 1) * size_old, factor * size_old, 0, size_old,
                            Hi1_cell.get_value())
    return Hii_new, Hi1_new


def transX2H(x, n):
    Hii = matrix_numpy(n, n, "float")
    Hi1 = matrix_numpy(n, n, "float")
    for i in range(n):
        Hii.set_block_value(i, i+1, 0, i+1, x[int(i*(i+1)/2): int((i+1)*(i+2)/2)])
        Hi1.set_block_value(i, i+1, 0, n, x[int(n*(n+1)/2+n*i): int(n*(n+1)/2+n*(i+1))])
    Hii_diag = Hii.diag()
    Hii = op.addmat(Hii, Hii.trans(), Hii_diag.nega())
    return Hii, Hi1


def transH2X(Hii, Hi1):
    n = Hii.get_size()[0]
    x = np.zeros(int((3 * n ** 2 + n) / 2))
    for i in range(n):
        x[int(i*(i+1)/2): int((i+1)*(i+2)/2)] = Hii.get_value(i, i + 1, 0, i + 1)
        x[int(n * (n + 1) / 2 + n * i): int(n * (n + 1) / 2 + n * (i + 1))] = Hi1.get_value(i, i+1, 0, n)
    return x


def objective(x_init, size, k_points, L, E_list_ref):
    Hii, Hi1 = transX2H(x_init, size)
    E_list = []
    for i in range(len(k_points)):
        H_temp = op.addmat(Hii, op.scamulmat(np.exp(-k_points[i] * L * 1j), Hi1.trans()),
                           op.scamulmat(np.exp(k_points[i] * L * 1j), Hi1))
        E_k = H_temp.eigenvalue().get_value().real
        E_list.append(E_k[:])
    diff = np.asarray(E_list) - E_list_ref
    # diff = diff / E_list_ref
    value = np.linalg.norm(diff, ord='fro')
    print(value)
    return value


def H_renormalize(Hii_cell, Hi1_cell, x_init: np, k_points, L):
    size = Hii_cell.get_size()[0]
    Hii_double, Hi1_double = H_extendSize(Hii_cell, Hi1_cell, 2)
    k_points_double = k_points / 2
    L_double = L * 2
    E_list_len = size
    E_list_ref = []
    E_ref_index_start = int(size - E_list_len / 2)
    E_ref_index_end = int(size + E_list_len / 2)
    for i in range(len(k_points_double)):
        H_temp = op.addmat(Hii_double,
                           op.scamulmat(np.exp(-k_points_double[i] * L_double * 1j), Hi1_double.trans()),
                           op.scamulmat(np.exp(k_points_double[i] * L_double * 1j), Hi1_double))
        E_k = H_temp.eigenvalue().get_value().real
        E_list_ref.append(E_k[E_ref_index_start: E_ref_index_end])
    E_list_ref = np.asarray(E_list_ref)
    res = minimize(objective, x_init, method='nelder-mead', args=(size, k_points, L, E_list_ref),
                   options={'xatol': 1e-8, 'disp': True, 'adaptive': True})
    Hii_new, Hi1_new = transX2H(res.x, size)
    return Hii_new, Hi1_new


def opValueInit(Hii_cell, Hi1_cell, size):
    H_k0 = op.addmat(Hii_cell, Hi1_cell.trans(), Hi1_cell)
    H_kpi = op.addmat(Hii_cell, Hi1_cell.trans().nega(), Hi1_cell.nega())
    eigvalue0 = H_k0.eigenvalue().get_value()
    eigvalue1 = H_kpi.eigenvalue().get_value()
    eigvec0 = H_k0.eigenvec().get_value()
    eigvec1 = H_kpi.eigenvec().get_value()
    eigvalueTotalUnsorted = np.hstack((eigvalue0, eigvalue1))
    eigvecTotalUnsorted = np.hstack((eigvec0, eigvec1))

    sorted_index = np.argsort(eigvalueTotalUnsorted)
    eigvecTotal = np.copy(eigvecTotalUnsorted)
    eigvalueTotal = np.copy(eigvalueTotalUnsorted)
    for i in range(len(sorted_index)):
        eigvecTotal[:, i] = eigvecTotalUnsorted[:, sorted_index[i]]
        eigvalueTotal[i] = eigvalueTotalUnsorted[sorted_index[i]]
    index_start = int(0.5 * size)
    index_end = int(index_start + size)

    phi = matrix_numpy(size, size)
    phi.copy(eigvecTotal[:, index_start: index_end])
    Hii_new = op.trimatmul(phi, Hii_cell, phi, 'cnn')
    Hi1_new = op.trimatmul(phi, Hi1_cell, phi, 'cnn')
    x = transH2X(Hii_new, Hi1_new)
    return x


def opValueInit2(Hii_cell, Hi1_cell, size):
    Hii_double, Hi1_double = H_extendSize(Hii_cell, Hi1_cell, 2)
    H_k0 = op.addmat(Hii_double, Hi1_double.trans(), Hi1_double)
    eigvec0 = H_k0.eigenvec().get_value()
    index_start = int(0.5 * size)
    index_end = int(index_start + size)
    phi = matrix_numpy(size, size)
    phi.copy(eigvec0[:, index_start: index_end])
    phi = op.qr_decomp(phi)
    # print(op.matmulmat(phi.trans(), phi).get_value())
    Hii_new = op.trimatmul(phi, Hii_double, phi, 'cnn')
    Hi1_new = op.trimatmul(phi, Hi1_double, phi, 'cnn')
    x = transH2X(Hii_new, Hi1_new)
    return x


def compute_band(Hii, Hi1, L, k_points):
    band_list = []
    for i in range(len(k_points)):
        H_temp = op.addmat(Hii, op.scamulmat(np.exp(-k_points[i] * L * 1j), Hi1.dagger()),
                           op.scamulmat(np.exp(k_points[i] * L * 1j), Hi1))
        E_k = H_temp.eigenvalue().get_value().real
        band_list.append(E_k[:])
    band_list = np.asarray(band_list)
    return band_list


def renormal_SanchoRubio(Hii, Hi1, E, eta):
    size_H = Hii.get_size()[0]
    w = complex(E, eta)
    eye = matrix_numpy(size_H, size_H)
    eye.identity()
    g_0 = op.inv(op.addmat(op.scamulmat(w, eye), Hii.nega()))
    alpha = op.trimatmul(Hi1, g_0, Hi1, "nnn")
    beta = alpha.dagger()
    epsilon_s = op.addmat(Hii, op.trimatmul(Hi1, g_0, Hi1, "nnc"))
    epsilon = op.addmat(Hii, op.trimatmul(Hi1, g_0, Hi1, "nnc"), op.trimatmul(Hi1, g_0, Hi1, "cnn"))
    return alpha, beta, epsilon_s, epsilon


def dosOfKE_SanchoRubio(Hii, Hi1, E, eta, k, L):
    w = complex(E, eta)
    alpha, beta, epsilon_s, epsilon = renormal_SanchoRubio(Hii, Hi1, E, eta)
    Hii_new = epsilon
    Hi1_new = alpha
    H_temp = op.addmat(Hii_new, op.scamulmat(np.exp(-k * L * 1j), Hi1_new.dagger()),
                       op.scamulmat(np.exp(k * L * 1j), Hi1_new))
    size_H = Hii.get_size()[0]
    eye = matrix_numpy(size_H, size_H)
    eye.identity()
    GF = op.inv(op.addmat(op.scamulmat(w, eye), H_temp.nega()))
    dos = min(- 2 * GF.tre().imag, 10)
    # dos = - 2 * GF.tre().imag
    return dos


