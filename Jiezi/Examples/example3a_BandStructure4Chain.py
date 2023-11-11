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
from Jiezi.Physics import hamilton, band
from Jiezi.Graph import builder
import numpy as np
import matplotlib.pyplot as plt

# build CNT and its Hmiltonian matrix
cnt = builder.CNT(n=4, m=0, Trepeat=3, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
H.build_H()
H.build_S(base_overlap=0.018)
# compute subband and eigenvector on specific K point
E_subband, U = band.subband(H, k=0)

# compute the bottom of conduction band and the band gap
Ec, Eg = band.get_EcEg(H)
print("bottom of conduction band is:", Ec)
print("band gap is:", Eg)

# plot band structure on the given k path
k_total, energy_band = band.band_structure(H, -np.pi, np.pi, 0.1)
i = 0
for band_k in energy_band:
    k = np.ones(energy_band[0].get_size()) * k_total[i]
    i += 1
    # # normalization of energy value (real value divided by the hopping value 2.97)
    # plt.scatter(k, band_k.get_value() / 2.97)
    plt.scatter(k, band_k.get_value(), s=10, c='r')
plt.gca().set_aspect('equal', adjustable='box')
plt.ylim((-12, 12))
plt.show()

