# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder

cnt = builder.CNT(5, 5, 3, a_cc=1.44, nonideal=False)

cnt.construct()
cnt.data_print()
cnt.data_plot()
# print(cnt.get_coordinate())
