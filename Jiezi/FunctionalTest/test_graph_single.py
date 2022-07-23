
import sys

sys.path.append("../../../")
import Jiezi.Graph.builder as builder

cnt = builder.CNT(4, 4, 1, a_cc=1.44, nonideal=False)
cnt.construct()
cnt.data_plot()

