import sys, os
script_path = os.path.dirname(__file__)
myutils_path = os.path.join(script_path, '../../../')
sys.path.append(myutils_path)

import matplotlib.pyplot as plt

from Jiezi import Graph
#import Graph

cnt = Graph.builder.CNT(4, 2, 3, nonideal=False)
cnt.construct()
cnt.data_print()
cnt.data_plot()

