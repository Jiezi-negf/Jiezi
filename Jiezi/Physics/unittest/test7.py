import unittest

import sys

sys.path.append("../../../")

from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder
from Jiezi.Physics.band import subband
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.rgf import rgf
from Jiezi.Physics.common import *
import numpy as np
from Jiezi.Physics import surface_gf
import matplotlib.pyplot as plt


class Test7(unittest.TestCase):

    def test_recursive_GF(self):
        cnt = builder.CNT(n=6, m=3, Trepeat=6, nonideal=False)
        cnt.construct()
        H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)

        E_subband, U = subband(H, k=0)
        # nm = 30
        # Hii, Hi1, Sii, form_factor = mode_space(H, U, 16)
        Hii = H.get_Hii()
        Hi1 = H.get_Hi1()
        Sii = H.get_Sii()
        E_list = [-0.3]

        ee = 0
        eta_rgf = 5e-6

        w = complex(E_list[ee], 0.0)
        sigma_ph = []
        nz = len(Hii)
        nm = Hii[0].get_size()[0]
        for i in range(len(E_list)):
            sigma_ph_ee = []
            for j in range(nz):
                sigma_ph_element = matrix_numpy(nm, nm)
                sigma_ph_ee.append(sigma_ph_element)
            sigma_ph.append(sigma_ph_ee)

        # this is the output of rgf method
        G_R_rgf, G_lesser, G_greater, G1i_lesser, Sigma_left_lesser, Sigma_left_greater, \
        Sigma_right_lesser, Sigma_right_greater = \
            rgf(ee, E_list, eta_rgf, mul, mur, Hii, Hi1, Sii, sigma_ph, sigma_ph)

        self.assertAlmostEqual(G_R_rgf[0].get_value(1, 1), 0.0011413888485991975-0.042457652641137066j)
