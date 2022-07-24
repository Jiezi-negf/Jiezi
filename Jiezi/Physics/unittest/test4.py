import unittest

import sys

sys.path.append("../../../")

from Jiezi.Graph import builder
from Jiezi.Physics import hamilton
from Jiezi.Physics.band import subband
from Jiezi.Physics.modespace import mode_space
from Jiezi.Physics.common import *



class Test4(unittest.TestCase):

    def test_mode_space(self):
        # construct the structure
        cnt = builder.CNT(n=4, m=4, Trepeat=3, nonideal=False)
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
        Hii_new, Hi1_new, Sii_new, form_factor = mode_space(H, U, nm - 6)
        self.assertAlmostEqual(Hii_new[0].get_value()[0][0].real, -4.2646731359046255)
        self.assertAlmostEqual(Hi1_new[0].get_value()[0][0].real, -1.3282243786348751)
        self.assertAlmostEqual(Sii_new[0].get_value()[0][0].real, 1.0241495341569977)
        self.assertEqual(Hii_new[0].get_size(), (10, 10))
        self.assertEqual(Hi1_new[0].get_size(), (10, 10))
        self.assertEqual(Sii_new[0].get_size(), (10, 10))




