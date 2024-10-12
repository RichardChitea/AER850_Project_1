import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

#Reading csv into dataframe
df = pd.read_csv("Project_1_Data.csv")

#print(df.info())

# Histograms
df.hist()

#3d plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for grp_name, grp_idx in df.groupby('Step').groups.items():
    x = df.iloc[grp_idx,0]
    y = df.iloc[grp_idx,1]
    z = df.iloc[grp_idx,2]
    ax.scatter3D(x, y, z, label=grp_name)
ax.legend(bbox_to_anchor=(1.2,0.5),loc='center left',frameon=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

#Attempting to find other patterns
attributes = ["X","Y","Z","Step"]
scatter_matrix(df[attributes])

#Using Pearson's r
corr_matrix = df.corr()
print(corr_matrix["Step"].sort_values(ascending=False))

#
a_columns = ['X',
             'Y',
             'Z']
b_columns = ['Step']

a = df[a_columns] 
b = df[b_columns]

