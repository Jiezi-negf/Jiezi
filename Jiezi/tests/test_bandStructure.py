# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import os
import sys
import numpy as np
import unittest

sys.path.append(os.path.abspath(__file__ + "/../../.."))
from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder

class TestBandStructure(unittest.TestCase):

    def test_band(self):
        cnt = builder.CNT(n=4, m=4, Trepeat=6, nonideal=False)
        cnt.construct()
        H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
        H.build_H()
        H.build_S(base_overlap=0.018)
        k_total, band_ = band.band_structure(H, 0, 3 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 20)
        np.testing.assert_array_almost_equal(band_[0].real().get_value(), TestBandStructure.expected_band0)
        np.testing.assert_array_almost_equal(band_[1].real().get_value(), TestBandStructure.expected_band1)


    expected_band0 = np.array([-8.71916509, -8.17799343, -8.17799343, -6.65333051,
                               -6.65333051, -4.53634086, -4.53634086, -3.19253438,
                               2.73930754, 4.20829463, 4.20829463, 6.62788929,
                               6.62788929, 8.45571327, 8.45571327, 9.12262156])

    expected_band1 = np.array([-8.71123521, -8.17026802, -8.17026802, -6.64605017, -6.64605017,
                               -4.52900812, -4.52900812, -3.18403372, 2.73017316, 4.20014166,
                               4.20014166, 6.6193373, 6.6193373, 8.4462631, 8.4462631,
                               9.11277876])
if __name__ == "__main__":
    unittest.main()