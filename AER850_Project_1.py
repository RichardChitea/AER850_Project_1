import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
# from pandas.plotting import scatter_matrix
import seaborn as sns

#Reading csv into dataframe
df = pd.read_csv("Project_1_Data.csv")
df = df.dropna()
df = df.drop(513)
df = df.reset_index(drop=True)
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

#Scatter Matrix Plot
# attributes = ["X","Y","Z","Step"]
# scatter_matrix(df[attributes])


#Using Pearson's r
pearson_matrix = df.corr()
print(pearson_matrix["Step"].sort_values(ascending=False))


#
a_columns = ['X',
             'Y',
             'Z']
b_columns = ['Step']

a = df[a_columns] 
b = df[b_columns]

#Stratified Sampling
df["Z_cat"] = pd.cut(df['Z'], bins = [0,2,4,6,np.inf], labels = [1,2,3,4])

my_splitter = StratifiedShuffleSplit(n_splits=1,
                                     test_size=0.2,
                                     random_state=42)

for train_index, test_index in my_splitter.split(df,df["Z_cat"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True) # set new indicies to be in order again
    strat_df_test = df.loc[test_index].reset_index(drop=True)

strat_df_train = strat_df_train.drop(columns=["Z_cat"],axis=1)
strat_df_test = strat_df_test.drop(columns=["Z_cat"],axis=1)

#print(df.columns)


#Variable Selection
A_train = strat_df_train.drop("Step", axis = 1)
B_train = strat_df_train["Step"]
A_test = strat_df_test.drop("Step", axis = 1)
B_test = strat_df_test["Step"]


#Correlation Matrix
corr_matrix = A_train.corr()
sns.heatmap(np.abs(corr_matrix))

corr1 = B_train.corr(A_train['X'])
print(corr1)
corr2 = B_train.corr(A_train['Y'])
print(corr2)
corr3 = B_train.corr(A_train['Z'])
print(corr3)


#Support vector machine (Linear, polynomial, rbf, and sigmoid)
svr = SVR()
param_grid_svr = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_svr.fit(A_train, B_train)
best_model_svr = grid_search_svr.best_estimator_
print("Best SVM Model:", best_model_svr)


# Training and Testing error for SVM
B_train_pred_svr = best_model_svr.predict(A_train)
B_test_pred_svr = best_model_svr.predict(A_test)
mae_train_svr = mean_absolute_error(B_train, B_train_pred_svr)
mae_test_svr = mean_absolute_error(B_test, B_test_pred_svr)
print(f"SVM - MAE (Train): {mae_train_svr}, MAE (Test): {mae_test_svr}")





