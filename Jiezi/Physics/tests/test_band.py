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


cnt = builder.CNT(n=5, m=5, Trepeat=6, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
k_total, band = band.band_structure(H, 0, 3 * 3.14 / 1.44, 1 * 3.14 / 1.44 / 20)
print(band[0].get_value())
i = 0
for band_k in band:
    k = np.ones(band[0].get_size()) * k_total[i]
    i += 1
    plt.scatter(k, band_k.get_value() / 2.97)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
