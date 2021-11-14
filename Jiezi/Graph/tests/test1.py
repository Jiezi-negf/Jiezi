# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

from Jiezi import Graph
import sys, os
script_path = os.path.dirname(__file__)
myutils_path = os.path.join(script_path, '../../../')
sys.path.append(myutils_path)

cnt = Graph.builder.CNT(4, 2, 3, a_cc=1.44, onsite=-0.28, hopping=-2.97, nonideal=False)
cnt.construct()
cnt.data_print()
cnt.data_plot()
print(cnt.get_onsite_value(), cnt.get_hopping_value())

