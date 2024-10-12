import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Reading csv into dataframe
df = pd.read_csv("Project_1_Data.csv")

#print(df.info())

# Histograms
df.hist()

#dfnp = df.to_numpy()

#3d plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df['X'],df['Y'],df['Z'])
plt.show()


a_columns = ['X',
             'Y',
             'Z']
b_columns = ['Step']

a = df[a_columns] 
b = df[b_columns]

