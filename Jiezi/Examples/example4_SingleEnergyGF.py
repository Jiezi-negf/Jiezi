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
from Jiezi.Physics import hamilton, rgf, band
from Jiezi.Graph import builder
from Jiezi.Physics.common import *
from Jiezi.LA.matrix_numpy import matrix_numpy

# build CNT and its hamiltonian matrix
cnt = builder.CNT(n=4, m=0, Trepeat=3, nonideal=False)
cnt.construct()
mul = 0.0
mur = 0.0
H = hamilton.hamilton(cnt, onsite=0.0, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)

# get useful matrix
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
Sii = H.get_Sii()
S00 = H.get_S00()
lead_H00_L, lead_H00_R = H.get_lead_H00()
lead_H10_L, lead_H10_R = H.get_lead_H10()

# compute subband to determine the energy range
E_subband, U = band.subband(H, k=0)
# pick up the min and max value of E_subband
nz = cnt.get_Trepeat()
nm = Hii[0].get_size()[0]
min_temp = []
max_temp = []
for energy in E_subband:
    min_temp.append(energy.get_value()[0])
    max_temp.append(energy.get_value()[-1])
min_subband = min(min_temp).real
max_subband = max(max_temp).real
# define Energy list that should be computed
E_start = min(mul, mur, min_subband) - 10 * KT + 7
E_end = max(mul, mur, max_subband) + 10 * KT - 6.5
E_step = 0.001
E_list = np.arange(E_start, E_end, E_step)
print("Energy list:", E_start, "to", E_end, "Step of energy:", E_step)

# define the phonon self-energy matrix as zero matrix
sigma_r_ph = []
sigma_lesser_ph = []
for ee in range(len(E_list)):
    sigma_ph_ee = []
    for j in range(nz):
        sigma_ph_element = matrix_numpy(nm, nm)
        sigma_ph_ee.append(sigma_ph_element)
    sigma_r_ph.append(sigma_ph_ee)
    sigma_lesser_ph.append(sigma_ph_ee)

# set parameters and compute the Green's functions by RGF method
ee = 10
eta = 5e-6
G_R, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
    Sigma_right_lesser, Sigma_right_greater = \
    rgf.rgf(ee, E_list, eta, mul, mur, Hii, Hi1, Sii, S00,
        lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
        sigma_lesser_ph, sigma_r_ph)
