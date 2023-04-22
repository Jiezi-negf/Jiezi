# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================


import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../../../")

from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder


cnt = builder.CNT(n=4, m=0, Trepeat=18, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
total_neighbor = cnt.get_total_neighbor()
layertolayer = cnt.get_layertolayer()
cell_start = 1
cell_repeat = 10
k_total, energy_band = band.band_structure_defect(H, 0, 6 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 20,
                                                  cell_start, cell_repeat)
# Hii_defect, Hi1_defect, Sii_defect, Si1_defect = H.H_defect_band([5], -5, 3)
H.H_add_defect([37], -100)
Hii_defect, Hi1_defect, Sii_defect, Si1_defect = H.H_defect_band(cell_start, cell_repeat)
print(band.get_EcEg(H))
# k_total, energy_band = band.band_structure(H, 0, 6 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 20)

k_total_defect, energy_band_defect = band.band_structure_defect(H, 0, 6 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 20,
                                           cell_start, cell_repeat)

# print(energy_band[0].get_value())
i = 0
for band_k in energy_band:
    k = np.ones(energy_band[0].get_size()) * k_total[i]
    i += 1
    # # normalization of energy value (real value divided by the hopping value 2.97)
    # plt.scatter(k, band_k.get_value() / 2.97)
    # plt.subplot(1, 2, 1)
    plt.scatter(k, band_k.get_value(), s=10, c='r')

i = 0
for band_k in energy_band_defect:
    k = np.ones(len(band_k.get_value())) * k_total_defect[i]
    i += 1
    # # normalization of energy value (real value divided by the hopping value 2.97)
    # plt.scatter(k, band_k.get_value() / 2.97)
    # plt.subplot(1, 2, 2)
    plt.scatter(k, band_k.get_value(), s=5, c='g')
plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((-12, 12))
plt.show()
