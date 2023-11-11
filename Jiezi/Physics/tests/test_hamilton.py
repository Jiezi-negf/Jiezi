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
from Jiezi.LA import operator as op
import time

cnt = builder.CNT(n=4, m=2, Trepeat=5000, nonideal=False)
cnt.construct()
for i in range(20):
    H = hamilton.hamilton(cnt, onsite=-0.28, hopping=-2.97)
    H.build_H()
    H.build_S(base_overlap=0.018)
    time.sleep(5)

# Hii = H.get_Hii()
# Hi1 = H.get_Hi1()
# Hi1[2].copy(op.scamulmat(10000, Hi1[2]).get_value())
# print(H.get_Hii()[2].get_value()[0][47])
# print(H.get_Sii()[0].get_value())
