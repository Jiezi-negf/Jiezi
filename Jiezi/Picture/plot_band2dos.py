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
from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder
import numpy as np
import matplotlib.pyplot as plt

# build CNT and its Hamiltonian matrix
nz = 60
cnt = builder.CNT(n=5, m=5, Trepeat=nz, nonideal=False)
cnt.construct()
L = cnt.get_singlecell_length()
H = hamilton.hamilton(cnt, onsite=0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
Hii = H.get_Hii()[1]
Hi1 = H.get_Hi1()[1]

# set k points and compute band
dense = 1000
k_points = np.arange(-np.pi / L, np.pi / L, 2*np.pi / (nz * L) / dense)
bandArray = hamilton.compute_band(Hii, Hi1, L, k_points)

# set energy domain of interest
E_start = -8
E_end = 8
E_step = 0.001
E_list = np.arange(E_start, E_end, E_step)
dos0 = band.band2dos(bandArray, E_list)
# nz is divided because dos is the LDOS(one single layer)
# E_step is divided because entry of dos0 is the states amount in energy scope which has the length E_step
# the dos is the density on E, so the energy step need to be divided
dos = list(dos0[:, 1] / (dense * nz * E_step))

# plot dos of E
plt.plot(dos0[:, 0], dos)
plt.show()

# write it to file
num_Epoints = len(E_list)
dataXY = np.zeros((num_Epoints, 2))
dataXY[:, 0:1] = np.asarray(E_list).reshape((num_Epoints, 1))
dataXY[:, 1:2] = np.asarray(dos).reshape((num_Epoints, 1))
path = os.path.abspath(__file__ + "/../../Files/plot")
fileName = "/band2dos55.dat"
fname = path + fileName
np.savetxt(fname, dataXY, fmt='%.18e', delimiter=' ', newline='\n')