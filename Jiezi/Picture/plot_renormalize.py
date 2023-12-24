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
print(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Physics.common import *
import numpy as np
from Jiezi.Physics import hamilton
from Jiezi.Graph import builder

cnt = builder.CNT(n=10, m=0, Trepeat=6, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
Hii_cell = H.get_Hii()[1]
Hi1_cell = H.get_Hi1()[1]
size = Hii_cell.get_size()[0]
L = 1
# k_points = np.arange(-np.pi / L, np.pi / L, 0.05)

# ***** renormalize based on sancho-rubio *****
eta = 5e-6
num_k = 100
num_E = 1000
k_list = np.linspace(-np.pi / L, np.pi / L, num_k)
E_list = np.linspace(-6, 6, num_E)
dos = np.zeros((num_k, num_E))
iter_n = 1

# plot dos of k and E which is computed from G(k, E)
for i, k in enumerate(k_list):
    for j, E in enumerate(E_list):
        Hi1_new, beta, epsilon_s, Hii_new = hamilton.renormal_SanchoRubio(Hii_cell, Hi1_cell, E, eta, iter_n)
        dos[i, j] = hamilton.dosOfKE_SanchoRubio(Hii_new, Hi1_new, E, eta, k, L)
path_Files = os.path.abspath(os.path.join(__file__, "../..", "Files"))
fileName = "/renormal2.dat"
fname = path_Files + fileName
np.savetxt(fname, dos, fmt='%.18e', delimiter=' ', newline='\n')
