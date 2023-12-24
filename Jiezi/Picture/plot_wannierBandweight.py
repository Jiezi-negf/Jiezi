# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import os
import sys
sys.path.append(os.path.abspath(__file__ + "/../../.."))
import math
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.LA.vector_numpy import vector_numpy
from Jiezi.LA import operator as op
import numpy as np
from Jiezi.Physics.common import ifdagger
from Jiezi.Physics import w90_trans as w90


path_Files = "/home/zjy/Jiezi/Jiezi/Files/"
path_hr = path_Files + "wannier90_hr_new.dat"
r_set, hr_set = w90.read_hamiltonian(path_hr)

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
    # swap for new version
    swap_list = [(3, 55), (13, 56), (15, 41), (35, 43)]
    for j in range(len(swap_list)):
        hr_temp = hr_temp.swap_index(swap_list[j][0], swap_list[j][1])
    ## swapped matrix
    hr_cnt_set[i].copy(hr_temp.get_value(40, 72, 40, 72))
    hr_graphene_set[i].copy(hr_temp.get_value(0, 40, 0, 40))
    hr_total_set[i].copy(hr_temp.get_value(0, 72, 0, 72))

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
# read k-points from .kpt
kptFilePath = path_Files + "wannier90_band.kpt"
kptList = w90.read_kpt(kptFilePath)
num_kpt = len(kptList)
# considering neighbor to different level by deleting different index in r_set
threshold_index = 21
deleted_index = []
for i in range(len(r_set)):
    if abs(r_set[i][2]) > threshold_index:
        deleted_index.append(i)
for i in range(len(deleted_index)):
    r_set.pop(deleted_index[len(deleted_index) - i - 1])
    hr_total_set.pop(deleted_index[len(deleted_index) - i - 1])
# compute all the eigenvalues on each k-point in k-path
matrixWholeEK = w90.computeEKonKpath(kptList, hr_total_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3).get_value()


# compute all the weight
num_band = hr_set[0].get_size()[0]
num_kpt = len(kptList)
matrixWholeWeight = matrix_numpy(num_kpt, num_band)
for i in range(num_kpt):
    k = kptList[i]
    eigen_energy_k, eigen_vec_k = w90.subband_k(hr_set, r_set, r_1, r_2, r_3, k_1, k_2, k_3, k)
    for j in range(num_band):
        weight = np.linalg.norm(eigen_vec_k.get_value()[0:40, j:j+1], ord='fro') / np.linalg.norm(
            eigen_vec_k.get_value()[0:, j:j + 1], ord='fro')
        matrixWholeWeight.set_value(i, j, weight)
matrixWholeWeight = matrixWholeWeight.get_value()

axis_X = np.linspace(0, 3.8, num_kpt)
dataXYY = np.zeros((num_kpt * num_band, 3))
for i in range(num_kpt):
    for j in range(num_band):
        dataXYY[i * num_band + j, 0] = axis_X[i]
        dataXYY[i * num_band + j, 1] = matrixWholeEK[i, j]
        dataXYY[i * num_band + j, 2] = matrixWholeWeight[i, j]

path = "/home/zjy/Documents/picture4CPC"
fileName = "/w90bandWeight.dat"
fname = path + fileName
np.savetxt(fname, dataXYY, fmt='%.18e', delimiter=' ', newline='\n')