# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys

sys.path.append("../../../")

from Jiezi.Physics import hamilton
from Jiezi.Graph import builder
from Jiezi.NEGF.tests.fake_potential import fake_potential
# construct the structure
T_repeat = 10
cnt = builder.CNT(4, 2, T_repeat, onsite=-0.28, hopping=-2.97, nonideal=False)
cnt.construct()
H_cell = cnt.get_hamilton_cell()
H_hopping = cnt.get_hamilton_hopping()
nn = H_cell.shape[0]
hopping_value = cnt.get_hopping_value()
H = hamilton.hamilton(H_cell, H_hopping, nn, T_repeat)
H.build_H()
H.build_S(hopping_value, base_overlap=0.018)
Hii = H.get_Hii()
Hi1 = H.get_Hi1()
Sii = H.get_Sii()
fake_potential()
