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
# export the entries of (hr[0,0,0] - hr[0,0,10])
hr_op = hr_total_set
num_blocks = 11
num_entries = (hr_op[0].get_size()[0])**2
dataXY = np.zeros((num_blocks*num_entries, 2))
for i in range(num_blocks):
    axis_X = ((i+1) * np.ones(num_entries)).reshape((num_entries, 1))
    axis_Y = np.absolute(hr_op[i + 31].real().get_value()).reshape((num_entries, 1)) + \
         (1e-5 * np.ones(num_entries)).reshape((num_entries, 1))
    # axis_Y = np.log(axis_Y)
    dataXY[i*num_entries:(i+1)*num_entries, 0:1] = axis_X
    dataXY[i*num_entries:(i+1)*num_entries, 1:2] = axis_Y
path = "/home/zjy/Documents/picture4CPC"
fileName = "/w90entry.dat"
fname = path + fileName
np.savetxt(fname, dataXY, fmt='%.18e', delimiter=' ', newline='\n')