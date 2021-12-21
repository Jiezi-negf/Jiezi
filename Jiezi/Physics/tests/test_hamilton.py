# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys
import os
import numpy as np

sys.path.append("../../../")
from Jiezi.Physics import hamilton
from Jiezi.Graph import builder

cnt = builder.CNT(n=4, m=2, Trepeat=3, nonideal=False)
cnt.construct()
H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
phi_list = [1, 2, 3]
H.build_H(phi_list)
H.build_S(base_overlap=0.018)
print(H.get_Hii()[2].get_value())
print(H.get_Si1()[0].get_value())
