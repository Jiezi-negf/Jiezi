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
from Jiezi.Physics.w90_trans import read_hamiltonian, latticeVectorInit, w90_supercell_matrix
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.Physics import hamilton
from Jiezi.Graph import builder


# read hamiltonian matrix information from wannier90_hr_new.dat
path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
path_hr = path_Files + "/wannier90_hr_new.dat"
r_set, hr_set = read_hamiltonian(path_hr)
# sawp index to get the right order
# extract hamiltonian matrix of CNT
hr_cnt_set = [None] * len(hr_set)
hr_total_set = [None] * len(hr_set)
swap_list = [(3, 55), (13, 56), (15, 41), (35, 43)]
for i in range(len(hr_set)):
    hr_cnt_set[i] = matrix_numpy()
    hr_total_set[i] = matrix_numpy()
    hr_temp = matrix_numpy()
    hr_temp.copy(hr_set[i].get_value())
    for j in range(len(swap_list)):
        hr_temp = hr_temp.swap_index(swap_list[j][0], swap_list[j][1])
    ## swapped matrix
    hr_cnt_set[i].copy(hr_temp.get_value(40, 72, 40, 72))
    hr_total_set[i].copy(hr_temp.get_value(0, 72, 0, 72))
r_1, r_2, r_3, k_1, k_2, k_3 = latticeVectorInit()
k_coord = [0.33333, 0, 0]
num_supercell = 1
Mii_total, Mi1_total = w90_supercell_matrix(r_set, hr_total_set, r_1, r_2, r_3, k_1, k_2, k_3,
                                            k_coord, num_supercell)
Mii_cnt, Mi1_cnt = w90_supercell_matrix(r_set, hr_cnt_set, r_1, r_2, r_3, k_1, k_2, k_3,
                                        k_coord, num_supercell)
# build hamilton matrix
cnt = builder.CNT(n=8, m=0, Trepeat=60, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
H.H_readw90(Mii_total, Mi1_total, Mii_cnt, Mi1_cnt, num_supercell)
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
LEAD_H00_L, LEAD_H00_R = H.get_lead_H00()
LEAD_H01_L, LEAD_H01_R = H.get_lead_H10()
print("channel layer hamiltonian matrix Hii from wannier90 is:", Hii[1].get_value())
print("channel coupling hamiltonian matrix Hi1 from wannier90 is:", Hi1[1].get_value())
print("lead layer hamiltonian matrix Hii from wannier90 is:", LEAD_H00_L.get_value())
print("lead coupling hamiltonian matrix Hi1 from wannier90 is:", LEAD_H01_L.get_value())
print("coupling matrix between lead and channel from wannier90 is:", Hi1[0].get_value())