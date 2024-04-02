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

# build CNT and its Hmiltonian matrix
cnt = builder.CNT(n=4, m=0, Trepeat=3, nonideal=False)
cnt.construct()
lengthSingleCell = cnt.get_singlecell_length()

H = hamilton.hamilton(cnt, onsite=0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.0)
Hii = H.get_Hii()[1]
Hi1 = H.get_Hi1()[1]
L = lengthSingleCell
nz = 60
k_points = np.arange(-np.pi / L, np.pi / L, 2*np.pi/nz/L/1000)
bandArray = hamilton.compute_band(Hii, Hi1, L, k_points)

num_kpoints = bandArray.shape[0]
num_bands = bandArray.shape[1]
k_points_X = k_points.reshape((num_kpoints, 1))
dataXY = np.zeros((num_kpoints, num_bands + 1))
dataXY[0:, 0:1] = k_points_X
dataXY[0:, 1:] = bandArray

start = -1
end = 2.6
step = 0.01
E_list = np.arange(start, end, step)
dos0 = band.band2dos(bandArray, E_list)
dos = list(dos0[:, 1] / (1000 * nz * step))

num_Epoints = len(E_list)


plt.plot(dos0[:, 0], dos)
plt.show()

# path = "/home/zjy/Documents/picture4CPC"
# fileName = "/band2dos80.dat"
# fname = path + fileName
# np.savetxt(fname, dataXY, fmt='%.18e', delimiter=' ', newline='\n')