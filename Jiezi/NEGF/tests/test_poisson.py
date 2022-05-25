# Copyright 2021 Jiezi authors.
#
# This file is part of Jiezi. It is subject to the license terms in the file
# LICENSE.md found in the top-level directory of this distribution. A list of
# Jiezi authors can be found in the file AUTHORS.md at the top-level directory
# of this distribution.
# ==============================================================================

from Jiezi.Graph.builder import CNT
from Jiezi.Physics.PrePoisson import PrePoisson
from Jiezi.Physics.poisson import poisson
cnt = CNT(4, 4, 3)
cnt.construct()
geo_para, path_xml = PrePoisson(cnt)
print(geo_para)
poisson(geo_para, path_xml, 1, 1)

