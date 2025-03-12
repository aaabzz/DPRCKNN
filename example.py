from DPRC import DensityPeakRegionCluster
import numpy as np
import matplotlib.pyplot as plt
from colormap import color_mapping



data = np.loadtxt('jain.txt', dtype=np.float32, delimiter='\t')
dprc = DensityPeakRegionCluster(n_clusters=2, DPR_visualizaiton=True, cluster_visualization=True)
labels = dprc.fit(data)
