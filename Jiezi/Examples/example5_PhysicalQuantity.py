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
from Jiezi.Physics import hamilton, rgf, band, modespace, SCBA, quantity
from Jiezi.Graph import builder
from Jiezi.Physics.common import *
from Jiezi.LA.matrix_numpy import matrix_numpy

# build CNT and its hamiltonian matrix
cnt = builder.CNT(n=4, m=0, Trepeat=3, nonideal=False)
cnt.construct()
radius_tube = cnt.get_radius()
length_single_cell = cnt.get_singlecell_length()
z_total = cnt.get_length()
volume_cell = math.pi * radius_tube ** 2 * length_single_cell
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

# compute the mode space basis to decrease the size of H
nn = E_subband[0].get_size()[0]
Hii_new, Hi1_new, Sii_new, form_factor, U_new = modespace.mode_space(H, U, nn)

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

# define parameters used in SCBA loop
iter_SCBA_max = 10
TOL_SCBA = 1e-100
ratio_SCBA = 0.5
eta = 5e-6
ac = 2.5
op = 1.0e8
v_s = 5.0e5
omega = 4
# Dac = ac ** 2 * KT / (cnt.get_mass_density() * v_s ** 2)
# Dop = (h_bar * op) ** 2 / (2 * cnt.get_mass_density() * omega * E_step)
Dac = 0
Dop = 0
G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE, \
    Sigma_left_lesser_fullE, Sigma_left_greater_fullE, Sigma_right_lesser_fullE, Sigma_right_greater_fullE = \
    SCBA.SCBA(E_list, iter_SCBA_max, TOL_SCBA, ratio_SCBA, eta, mul, mur, Hii, Hi1, Sii, S00,
         lead_H00_L, lead_H00_R, lead_H10_L, lead_H10_R,
         sigma_lesser_ph, sigma_r_ph, form_factor, Dac, Dop, omega)

# compute physical quantity
n_spectrum, p_spectrum = quantity.carrierSpectrum(E_list, G_lesser_fullE, G_greater_fullE, volume_cell)
dos = quantity.densityOfStates(E_list, G_R_fullE, volume_cell)