# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================
import math
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA import operator as op
import numpy as np
from Jiezi.Physics.common import ifdagger
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


# define three basic lattice vectors in real space and get the lattice vector
def lattice_vector(r: list, r_1: vector_numpy, r_2: vector_numpy, r_3: vector_numpy):
    lat_vec = op.addvec(op.scamulvec(r[0], r_1), op.scamulvec(r[1], r_2), op.scamulvec(r[2], r_3))
    return lat_vec


def subband_k(hr_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k):
    nn = hr_set[0].get_size()[0]
    hk = matrix_numpy(nn, nn)
    for i in range(len(r_set)):
        r_vec = lattice_vector(r_set[i], r_1, r_2, r_3)
        k_vec = lattice_vector(k, k_1, k_2, k_3)
        hk = op.addmat(hk, op.scamulmat(np.exp(1j * op.vecdotvec(k_vec.trans(), r_vec)), hr_set[i]))
    eigen_energy_k = hk.eigenvalue()
    return eigen_energy_k


def read_hamiltonian(path_hr):
    with open(path_hr, "r") as f:
        lines = f.readlines()
    num_wan = int(lines[1])
    num_rpts = int(lines[2])
    lines_dengeneracy = num_rpts // 15 + 1
    r_set = [list] * num_rpts
    hr_set = [None] * num_rpts
    for i in range(num_rpts):
        index_first_line = 3 + lines_dengeneracy + i * num_wan ** 2
        line_first = lines[index_first_line].split()
        r_set[i] = list(map(int, line_first[0: 3]))
        hamiltonian_R = matrix_numpy(num_wan, num_wan)
        for j in range(num_wan ** 2):
            line = lines[index_first_line + j].split()
            m = int(line[3]) - 1
            n = int(line[4]) - 1
            element_mn = float(line[5])
            hamiltonian_R.set_value(m, n, element_mn)
        hr_set[i] = hamiltonian_R
    return r_set, hr_set


def latticeVectorInit():
    r_1 = vector_numpy(3)
    r_2 = vector_numpy(3)
    r_3 = vector_numpy(3)
    r_1.set_value((0, 0), 24.6881123)
    r_2.set_value((1, 0), 40)
    r_3.set_value((2, 0), 4.2761526)
    k_1 = vector_numpy(3)
    k_2 = vector_numpy(3)
    k_3 = vector_numpy(3)
    k_1.set_value((0, 0), 2 * math.pi / 24.6881123)
    k_2.set_value((1, 0), 2 * math.pi / 40)
    k_3.set_value((2, 0), 2 * math.pi / 4.2761526)
    return r_1, r_2, r_3, k_1, k_2, k_3


# create supercell
def w90_supercell_matrix(r_set, hr_set, r_1, r_2, r_3, k_1, k_2, k_3, k_coord, num_cell):
    k = lattice_vector(k_coord, k_1, k_2, k_3)
    matrix_size = hr_set[0].get_size()[0]
    Mii = matrix_numpy(matrix_size * num_cell, matrix_size * num_cell)
    Mi1 = matrix_numpy(matrix_size * num_cell, matrix_size * num_cell)
    for i in range(num_cell):
        for j in range(num_cell):
            Mii_ij = matrix_numpy(matrix_size, matrix_size)
            Mi1_ij = matrix_numpy(matrix_size, matrix_size)
            for r_x_i in [-1, 0, 1]:
                r_coord = [r_x_i, 0, j - i]
                r = lattice_vector(r_coord, r_1, r_2, r_3)
                phase_factor = np.exp(complex(0.0, op.vecdotvec(k.trans(), r)))
                Mii_ij = op.addmat(Mii_ij,
                                     op.scamulmat(phase_factor,
                                                  hr_set[r_set.index([r_x_i, 0, j - i])])
                                     )
                if j - i < 1:
                    Mi1_ij = op.addmat(Mi1_ij,
                                       op.scamulmat(phase_factor,
                                                    hr_set[r_set.index([r_x_i, 0, j - i + num_cell])])
                                       )
            Mii.set_block_value(i * matrix_size, (i + 1) * matrix_size,
                                j * matrix_size, (j + 1) * matrix_size,
                                Mii_ij)
            Mi1.set_block_value(i * matrix_size, (i + 1) * matrix_size,
                                j * matrix_size, (j + 1) * matrix_size,
                                Mi1_ij)

    return Mii, Mi1


def plotBand4Supercell(r_set, hr_set, kz_list, r_1, r_2, r_3, k_1, k_2, k_3, num_cell):
    num_band = hr_set[0].get_size()[0] * num_cell
    num_kpt = len(kz_list)
    matrixEK = matrix_numpy(num_kpt, num_band)
    for i in range(num_kpt):
        k_z_i = kz_list[i]
        k_coord_superMatrix = [0.3333, 0, k_z_i]
        Mii, Mi1 = w90_supercell_matrix(r_set, hr_set, r_1, r_2, r_3, k_1, k_2, k_3, k_coord_superMatrix, num_cell)
        k_coord = [0, 0, k_z_i]
        k = lattice_vector(k_coord, k_1, k_2, k_3)
        r_coord = [0, 0, num_cell]
        r = lattice_vector(r_coord, r_1, r_2, r_3)
        phase_factor = np.exp(complex(0.0, op.vecdotvec(k.trans(), r)))
        H_k = op.addmat(op.scamulmat(1/phase_factor, Mi1.dagger()),
                        Mii,
                        op.scamulmat(phase_factor, Mi1)
                        )
        eigen_energy_k = H_k.eigenvalue()
        matrixEK.set_block_value(i, i+1, 0, num_band, eigen_energy_k)
    matrixEK = matrixEK.trans()
    x = np.linspace(0, 0.48978493E-02 * num_kpt, num_kpt)
    for i in range(num_band):
        y = matrixEK.get_value(i, i+1, 0, num_kpt)[0, :]
        plt.plot(x, y)
        ax = plt.gca()
        ax.set_aspect(0.1)
    plt.show()


# read k-points from wannier_band.kpt file
def read_kpt(path_kpt):
    with open(path_kpt, "r") as f:
        lines = f.readlines()
    num_kpt = int(lines[0])
    kptList = [list] * num_kpt
    for i in range(num_kpt):
        line = lines[i + 1].split()
        k_a, k_b, k_c = float(line[0]), float(line[1]), float(line[2])
        kptList[i] = [k_a, k_b, k_c]
    return kptList


# compute all the eigenvalues on each k-point in k-Path and store it in a matrix(num_kpt, num_band)
def computeEKonKpath(kptList, hr_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3):
    num_band = hr_set[0].get_size()[0]
    num_kpt = len(kptList)
    matrixWholeEK = matrix_numpy(num_kpt, num_band)
    for i in range(num_kpt):
        k = kptList[i]
        eigen_energy_k = subband_k(hr_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k)
        matrixWholeEK.set_block_value(i, i+1, 0, num_band, eigen_energy_k)
    return matrixWholeEK


# plot every band along k-path according to matrixWholeEK
def plotBandAlongKpath(matrixWholeEK):
    matrixWholeEK_t = matrixWholeEK.trans()
    num_band, num_kpt = matrixWholeEK_t.get_size()
    x = np.linspace(0, 3.8, num_kpt)
    for i in range(num_band):
        y = matrixWholeEK_t.get_value(i, i+1, 0, num_kpt)[0, :]
        plt.plot(x, y)
    plt.show()


# read band value from wannier90_band.dat
def plotBandByReadFile(path_band, num_band, num_kpt):
    with open(path_band, "r") as f:
        lines = f.readlines()
    x = np.linspace(0, 3.8, num_kpt)
    for i in range(num_band):
        y = [float] * num_kpt
        for j in range(num_kpt):
            y[j] = float(lines[i * (num_kpt + 1) + j].split()[1])
        plt.plot(x, y)
    plt.show()


# plot function for fold band
def plotBandbyFold(matrixWholeEK, x):
    import warnings
    warnings.filterwarnings("ignore")
    matrixWholeEK_t = matrixWholeEK.trans()
    num_band, num_kpt = matrixWholeEK_t.get_size()
    for i in range(num_band):
        y = matrixWholeEK_t.get_value(i, i + 1, 0, num_kpt)[0, :]
        plt.plot(x, y)


if __name__ == "__main__":
    path_Files = "/home/zjy/Jiezi/Jiezi/Files/"
    path_hr = path_Files + "wannier90_hr_new.dat"
    r_set, hr_set = read_hamiltonian(path_hr)

    # test if hr_set[i]^dagger equals hr_set[2*10-i]
    error = 0.0
    for i in range(9):
        addition = op.addmat(hr_set[i], hr_set[2*10 - i])
        error += ifdagger(addition)
    print(error)

    # extract hamiltonian matrix of CNT
    hr_cnt_set = [None] * len(hr_set)
    hr_graphene_set = [None] * len(hr_set)
    hr_total_set = [None] * len(hr_set)
    for i in range(len(hr_set)):
        hr_cnt_set[i] = matrix_numpy()
        hr_graphene_set[i] = matrix_numpy()
        hr_total_set[i] = matrix_numpy()
        hr_temp = matrix_numpy()
        hr_temp.copy(hr_set[i].get_value())
        # # swap_list = [(4, 56), (5, 42), (13, 70), (15, 54), (23, 53), (35, 41)]
        # swap_list = [(35, 41), (5, 42), (66, 50), (53, 71), (54, 55), (55, 15), (56, 4), (66, 50), (70, 13), (71, 23)]
        # for j in range(len(swap_list)):
        #     hr_temp = hr_temp.swap_index(swap_list[j][0], swap_list[j][1])

        # swap for new version
        swap_list = [(3, 55), (13, 56), (15, 41), (35, 43)]
        for j in range(len(swap_list)):
            hr_temp = hr_temp.swap_index(swap_list[j][0], swap_list[j][1])
        ## swapped matrix
        hr_cnt_set[i].copy(hr_temp.get_value(40, 72, 40, 72))
        hr_graphene_set[i].copy(hr_temp.get_value(0, 40, 0, 40))
        hr_total_set[i].copy(hr_temp.get_value(0, 72, 0, 72))
        ## un-swapped matrix
        # hr_cnt_set[i].copy(hr_set[i].get_value(40, 72, 40, 72))
    # print(hr_cnt_set[0].get_size())
    # # test if hr_set[i]^dagger equals hr_set[2*10-i]
    # error = 0.0
    # for i in range(9):
    #     addition = op.addmat(hr_cnt_set[i], hr_cnt_set[2*10 - i])
    #     error += ifdagger(addition)
    # print(error)



    r_1 = vector_numpy(3)
    r_2 = vector_numpy(3)
    r_3 = vector_numpy(3)
    r_1.set_value((0, 0), 24.6881123)
    r_2.set_value((1, 0), 40)
    r_3.set_value((2, 0), 4.2761526)
    k_1 = vector_numpy(3)
    k_2 = vector_numpy(3)
    k_3 = vector_numpy(3)
    k_1.set_value((0, 0), 2 * math.pi / 24.6881123)
    k_2.set_value((1, 0), 2 * math.pi / 40)
    k_3.set_value((2, 0), 2 * math.pi / 4.2761526)
    # k = [0.3333, 0, -0]
    # eigen_energy_k = subband_k(hr_cnt_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k)
    # eigen_energy_k = subband_k(hr_graphene_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k)
    # eigen_energy_k = subband_k(hr_graphene_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k)
    # print(eigen_energy_k.get_value())
    # print(eigen_energy_k.get_size())


    # # <<<<<<<<<<PART: plot superlattice's band<<<<<<<<<<<<<
    # # create supercell
    # num_cell = 2
    # # plot the band structure of supercell
    # import warnings
    # warnings.filterwarnings("ignore")
    # kz_start = -0.5 / num_cell
    # kz_list = np.linspace(kz_start, -kz_start, int(-2*kz_start/0.003333))
    # plotBand4Supercell(r_set, hr_graphene_set, kz_list, r_1, r_2, r_3, k_1, k_2, k_3, num_cell)



    # <<<<<<<<<<<<<<<<<PART: plot band based on single cell<<<<<<
    # read k-points from .kpt
    kptFilePath = path_Files + "wannier90_band.kpt"
    kptList = read_kpt(kptFilePath)
    num_kpt = len(kptList)
    # considering neighbor to different level by deleting different index in r_set
    threshold_index = 2
    deleted_index = []
    for i in range(len(r_set)):
        if abs(r_set[i][2]) > threshold_index:
            deleted_index.append(i)
    for i in range(len(deleted_index)):
        r_set.pop(deleted_index[len(deleted_index) - i - 1])
        hr_cnt_set.pop(deleted_index[len(deleted_index) - i - 1])
    # compute all the eigenvalues on each k-point in k-path
    matrixWholeEK = computeEKonKpath(kptList, hr_cnt_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3)
    # plot every band
    import warnings
    warnings.filterwarnings("ignore")
    plotBandAlongKpath(matrixWholeEK)

    # # <<<<<<<<<<<<PART: plot band from wannier90_band.dat<<<<<<<<<<<<<<<<
    # bandFilePath = path_Files + "wannier90_band_new.dat"
    # num_band = 72
    # plotBandByReadFile(bandFilePath, num_band, num_kpt)

    # # <<<<<<<<<<<PART: plot folded band of single lattice to form band of super lattice<<<<<<<<<<
    # kz_start = 0.5
    # kz_end = -0.5
    # kz_step = -0.005
    # kz_list = np.arange(kz_start, kz_end + kz_step, kz_step)
    # num_kpt = len(kz_list)
    # K2X_scale = -3
    # x_list = np.multiply(K2X_scale, kz_list)
    #
    # # plot band from -1/4 to 1/4
    # kpt_inter = []
    # x_inter = []
    # for i in range(int(num_kpt / 2)):
    #     index_start = int(num_kpt / 4)
    #     kpt_inter.append([0.3333, 0, kz_list[i + index_start]])
    #     x_inter.append(x_list[i + index_start])
    # matrixWholeEK = computeEKonKpath(kpt_inter, hr_graphene_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3)
    # plotBandbyFold(matrixWholeEK, x_inter)
    #
    # # fold band from (-1/2, -1/4) to (-1/4, 0)
    # kpt_left = []
    # x_left = []
    # for i in range(int(num_kpt / 4)):
    #     index_start = int(num_kpt / 4)
    #     kpt_left.append([0.3333, 0, kz_list[index_start - i]])
    #     x_left.append(x_list[i + index_start])
    # matrixWholeEK = computeEKonKpath(kpt_left, hr_graphene_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3)
    # plotBandbyFold(matrixWholeEK, x_left)
    #
    # # fold band from (1/4, 1/2) to (0, 1/2)
    # kpt_right = []
    # x_right = []
    # for i in range(int(num_kpt / 4)):
    #     index_start = int(num_kpt / 4 * 2)
    #     kpt_right.append([0.3333, 0, kz_list[num_kpt - i - 1]])
    #     x_right.append(x_list[i + index_start])
    # matrixWholeEK = computeEKonKpath(kpt_right, hr_graphene_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3)
    # plotBandbyFold(matrixWholeEK, x_right)
    # ax = plt.gca()
    # ax.set_aspect(0.1)
    # plt.show()
