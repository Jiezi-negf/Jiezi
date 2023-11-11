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
from Jiezi.Visualization.Data2File import atomPos2XYZ

cnt = builder.CNT(8, 0, 1, a_cc=1.42536)
cnt.construct()
cnt.data_print()
# cnt.data_plot()
path = '../../Files/'
atomPos2XYZ(cnt.get_coordinate(), path)

cnt.data_print()
print(cnt.get_coordinate())
print(cnt.get_nn())
print(cnt.get_radius())
print(cnt.get_singlecell_length())
print(cnt.get_length())
volume_cell = 3.14 * cnt.get_radius() ** 2 * cnt.get_singlecell_length()
# doping_line = 1
# doping_volume = doping_line * cnt.get_singlecell_length() / volume_cell
# print(doping_volume)

