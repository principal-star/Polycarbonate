# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:30:25 2024

@author: Dean FME
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#ds_fea0=pd.read_csv('d:/msk/lenin/vbalaji/4by4_dendexdertr.csv')
ds_fea0=pd.read_csv('d:/msk/lenin/polycarbonate/mldata_ka2.csv')
ds_fea1=ds_fea0.iloc[:,0:4]
ds_tar_end1=ds_fea0.iloc[:,4]#4 for end 5 for exd and 6 for erd 7 for tr

pa=['WP',	'TR',	'AMFR',	'SOD']
#grl=[0,1,2]
#grl=[0,1,3]
grl=[0,1,2]
"""
# Generating random data for demonstration
np.random.seed(0)
n_samples = 100
X1 = np.random.rand(n_samples) * 10
X2 = np.random.rand(n_samples) * 10
X3 = np.random.rand(n_samples) * 10
Y = X1 + X2 - X3 + np.random.randn(n_samples) * 2
"""
# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for the three independent variables and the dependent variable
ax.scatter(ds_fea1.iloc[:,grl[0]], ds_fea1.iloc[:,grl[1]], ds_fea1.iloc[:,grl[2]], c=ds_tar_end1, cmap='viridis', label='Data')

# Labels and title
ax.set_xlabel(pa[grl[0]])
ax.set_ylabel(pa[grl[1]])
ax.set_zlabel(pa[grl[2]])
ax.set_title('3D Scatter Plot of ' +  pa[grl[0]] +',' + pa[grl[1]] + ' and ' + pa[grl[2]] + '  vs KA ')

# Add color bar
cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
cbar.set_label('Dependent Variable (Y)')

plt.show()
fign3='d:/msk/lenin/polycarbonate/polycar_ka2_4dplot.jpg'    
fig.savefig(fign3)
