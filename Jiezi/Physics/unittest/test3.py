import unittest

import sys

sys.path.append("../../../")
from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder


class Test3(unittest.TestCase):

    def test_band(self):
        cnt = builder.CNT(n=4, m=4, Trepeat=6, nonideal=False)
        cnt.construct()
        H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)
        k_total, band_ = band.band_structure(H, 0, 3 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 20)
        self.assertAlmostEqual(band_[0].get_value()[0].real, -8.7191650853889868)
        self.assertAlmostEqual(band_[0].get_value()[15].real, 9.1226215644820332)
        self.assertAlmostEqual(band_[0].get_value()[0].imag, 0.0)
        self.assertAlmostEqual(band_[0].get_value()[15].imag, 0.0)
