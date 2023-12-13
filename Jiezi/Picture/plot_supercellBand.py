# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import os
import sys

sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from Jiezi.Physics.common import *
import numpy as np
import matplotlib.pyplot as plt

from Jiezi.Physics import hamilton
from Jiezi.Graph import builder

cnt = builder.CNT(n=4, m=0, Trepeat=6, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.0)
Hii_cell = H.get_Hii()[1]
Hi1_cell = H.get_Hi1()[1]
size = Hii_cell.get_size()[0]
L = 1
k_points = np.arange(-np.pi / L, np.pi / L, 0.05)

L_2 = 2 * L
L_4 = 4 * L
k_points_2 = k_points / 2
k_points_4 = k_points / 4

# plot band structure
Hii_double, Hi1_double = hamilton.H_extendSize(Hii_cell, Hi1_cell, 2)
band = hamilton.compute_band(Hii_double, Hi1_double, L_2, k_points_2)
num_kpoints = band.shape[0]
num_bands = band.shape[1]
k_points_X = k_points.reshape((num_kpoints, 1))
dataXY = np.zeros((num_kpoints, num_bands + 1))
dataXY[0:, 0:1] = k_points_X
dataXY[0:, 1:] = band
path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
fileName = "/40CNTbandsupercell2.dat"
fname = path_Files + fileName
np.savetxt(fname, dataXY, fmt='%.18e', delimiter=' ', newline='\n')
