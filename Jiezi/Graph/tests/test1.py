import sys, os
script_path = os.path.dirname(__file__)
myutils_path = os.path.join(script_path, '../../')
sys.path.append(myutils_path)

import matplotlib.pyplot as plt

import Graph

cnt = Graph.builder.CNT(3, 1, 1, nonideal=False)
cnt.construct()
cnt.data_print()
cnt.data_plot()

