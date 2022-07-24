import unittest

import sys

sys.path.append("../../../")

from Jiezi.Graph import builder
from Jiezi.NEGF.tests.fake_potential import fake_potential
from Jiezi.Physics import hamilton
from Jiezi.Physics.band import subband
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.rgf import rgf
from Jiezi.LA.matrix_numpy import matrix_numpy
from Jiezi.Physics.common import *
from Jiezi.Physics.quantity import quantity
import numpy as np
import matplotlib.pyplot as plt


class Test1(unittest.TestCase):
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
    Hii_new, Hi1_new, Sii_new, form_factor = mode_space(H, U, nm - 10)
    # Hii_new = H.get_Hii()
    # Hi1_new = H.get_Hi1()
    # Sii_new = H.get_Sii()

    # pick up the min and max value of E_subband
    min_temp = []
    max_temp = []
    for energy in E_subband:
        min_temp.append(energy.get_value()[0])
        max_temp.append(energy.get_value()[nm - 1])
    min_subband = min(min_temp).real
    max_subband = max(max_temp).real

    # define Energy list that should be computed
    start = min(mul, mur, min_subband) - 10 * KT
    end = max(mul, mur, max_subband) + 10 * KT
    step = 0.1
    E_list = np.arange(start, end, step)

    # compute GF by RGF iteration
    # define the phonon self-energy matrix as zero matrix
    eta = 5e-6
    sigma_ph = []
    nz = len(Hii_new)
    nm = Hii_new[0].get_size()[0]
    sigma_ph_element = matrix_numpy(nm, nm)
    for i in range(len(E_list)):
        sigma_ph_ee = []
        for j in range(nz):
            sigma_ph_ee.append(sigma_ph_element)
        sigma_ph.append(sigma_ph_ee)

    # initial list to store GF matrix of every energy
    G_R_fullE = []
    G_lesser_fullE = []
    G_greater_fullE = []
    G1i_lesser_fullE = []
    Sigma_left_lesser_fullE = []
    Sigma_left_greater_fullE = []
    Sigma_right_lesser_fullE = []
    Sigma_right_greater_fullE = []

    for ee in range(len(E_list)):
        G_R, G_lesser, G_greater, G1i_lesser, \
        Sigma_left_lesser, Sigma_left_greater, \
        Sigma_right_lesser, Sigma_right_greater = \
            rgf(ee, E_list, eta, mul, mur, Hii_new, Hi1_new, Sii_new, sigma_ph, sigma_ph)
        G_R_fullE.append(G_R)
        G_lesser_fullE.append(G_lesser)
        G_greater_fullE.append(G_greater)
        G1i_lesser_fullE.append(G1i_lesser)
        Sigma_left_lesser_fullE.append(Sigma_left_lesser)
        Sigma_left_greater_fullE.append(Sigma_left_greater)
        Sigma_right_lesser_fullE.append(Sigma_right_lesser)
        Sigma_right_greater_fullE.append(Sigma_right_greater)

    n_tol, p_tol, J, dos = quantity(E_list, G_R_fullE, G_lesser_fullE, G_greater_fullE, G1i_lesser_fullE,
                                    Sigma_left_lesser_fullE, Sigma_left_greater_fullE,
                                    Sigma_right_lesser_fullE, Sigma_right_greater_fullE,
                                    Hi1_new, volume_cell)

    def test_zigzagCNT_para(self):
        self.assertAlmostEqual(Test1.n_tol[0].real, 0.10819495065116673, 3)
        self.assertAlmostEqual(Test1.n_tol[1].real, 0.10819355968932076, 3)
        self.assertAlmostEqual(Test1.n_tol[2].real, 0.10819462535077344, 3)

        self.assertAlmostEqual(Test1.dos[0][1], 1.43172812692925e-13, 3)
        self.assertAlmostEqual(Test1.dos[0][2], 6.1581973527667403e-10, 3)
        self.assertAlmostEqual(Test1.dos[1][0], 6.5228097721722976e-10, 3)
