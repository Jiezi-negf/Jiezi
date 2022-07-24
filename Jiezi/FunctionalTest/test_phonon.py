# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys

sys.path.append("../../../")

from Jiezi.Graph import builder
from Jiezi.Physics import hamilton
from Jiezi.Physics.band import subband
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.SCBA import SCBA
from Jiezi.Physics.rgf import rgf
from Jiezi.Physics.common import *
from Jiezi.Physics.quantity import quantity
import numpy as np
import matplotlib.pyplot as plt

# construct the structure
cnt = builder.CNT(n=8, m=0, Trepeat=3, nonideal=False)
cnt.construct()
radius_tube = cnt.get_radius()
length_single_cell = cnt.get_singlecell_length()
volume_cell = math.pi * radius_tube ** 2 * length_single_cell


# build hamilton matrix
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)

# compute the eigenvalue(subband energy) and eigenvector(transformation matrix)
# E_subband is a list, the length of which is equal to number of layers.
# The length of E_subband[i] is equal to atoms in every single layer.
# U is a list, the length of which is equal to number of layers.
# The size of U[i] is equal to every single block matrix of H
E_subband, U = subband(H, k=0)

# compute the mode space basis to decrease the size of H
nm = E_subband[0].get_size()[0]
Hii_new, Hi1_new, Sii_new, form_factor = mode_space(H, U, nm-10)
# Hii_new = H.get_Hii()
# Hi1_new = H.get_Hi1()
# Sii_new = H.get_Sii()

# pick up the min and max value of E_subband
nm = Hii_new[0].get_size()[0]
min_temp = []
max_temp = []
for energy in E_subband:
    min_temp.append(energy.get_value()[0])
    max_temp.append(energy.get_value()[nm - 1])
min_subband = min(min_temp).real
max_subband = max(max_temp).real

# define Energy list that should be computed
E_start = min(mul, mur, min_subband) - 10 * KT
E_end = max(mul, mur, max_subband) + 10 * KT
E_step = 0.05
E_list = np.arange(E_start, E_end, E_step)
print(E_list)

# compute GF by RGF iteration
# define the phonon self-energy matrix as zero matrix
sigma_r_ph = []
sigma_lesser_ph = []
nz = len(Hii_new)
nm = Hii_new[0].get_size()[0]
for ee in range(len(E_list)):
    sigma_ph_ee = []
    for j in range(nz):
        sigma_ph_element = matrix_numpy(nm, nm)
        sigma_ph_ee.append(sigma_ph_element)
    sigma_r_ph.append(sigma_ph_ee)
    sigma_lesser_ph.append(sigma_ph_ee)

# initial list to store GF matrix of every energy
iter_max = 10
TOL = 1e-100
ratio = 0.5
eta = 5e-6
ac = 2.5
op = 1.0e8
v_s = 5.0e5
omega = 4
Dac = ac ** 2 * KT / (cnt.get_mass_desity() * v_s ** 2)
Dop = (h_bar * op) ** 2 / (2 * cnt.get_mass_desity() * omega * E_step)
G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, \
    Sigma_left_lesser_fullE, Sigma_left_greater_fullE, Sigma_right_lesser_fullE, Sigma_right_greater_fullE = \
SCBA(E_list, iter_max, TOL, ratio, eta, mul, mur, Hii_new, Hi1_new, Sii_new,
            sigma_lesser_ph, sigma_r_ph, form_factor, Dac, Dop, omega)

n_tol, p_tol, J, dos = quantity(E_list, G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE,
                           Sigma_left_lesser_fullE, Sigma_left_greater_fullE,
                           Sigma_right_lesser_fullE, Sigma_right_greater_fullE,
                           Hi1_new, volume_cell)
print(J)
x = range(len(J))
plt.plot(x, J)
plt.show()
