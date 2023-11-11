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
from Jiezi.Visualization.Data2File import atomPos2XYZ
from Jiezi.Graph import builder
from Jiezi.Visualization.Visualization_Graph import visual

cnt80 = builder.CNT(8, 0, 1, a_cc=1.42536)
cnt80.construct()
cnt80.data_print()

cnt42 = builder.CNT(4, 2, 3, a_cc=1.42536)
cnt42.construct()
cnt42.data_print()

# save the atom coordinates as .xyz format to be opened by VESTA for visualization
path_xyz = os.path.abspath(os.path.join(__file__, "../..", "Files"))
atomPos2XYZ(cnt80.get_coordinate(), path_xyz)

# if the mayavi module has been installed, you can use it to visualization without export .xyz file
visual(cnt80)
