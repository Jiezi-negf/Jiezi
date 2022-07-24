import unittest

import sys

sys.path.append("../../../")
from Jiezi.LA.vector_numpy import vector_numpy
import numpy as np
from Jiezi.Physics import hamilton, band, modespace, surface_gf
from Jiezi.Graph import builder
from Jiezi.LA import operator as op


class Test8(unittest.TestCase):

    def test_surface_GF(self):
        cnt = builder.CNT(n=5, m=5, Trepeat=3, nonideal=False)
        cnt.construct()
        H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)
        E_subband, U = band.subband(H, k=0)

        E_list = [0]
        Hii = H.get_Hii()
        Hi1 = H.get_Hi1()
        Sii = H.get_Sii()
        nz = len(Hii)
        nm = Hii[0].get_size()[0]
        # Hii, Hi1, Sii, form_factor = modespace.mode_space(H, U, nm)
        ee = 0
        eta = 5e-6
        GBB_L = surface_gf.surface_gf(E_list[ee], eta, Hii[0], Hi1[0].dagger(), Sii[0], TOL=1e-12)[1]
        self.assertAlmostEqual(GBB_L.get_value(0, 0), -0.017229597555830024-0.038995030532094216j, 3)
