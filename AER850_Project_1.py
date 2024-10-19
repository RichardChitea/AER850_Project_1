import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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
print(f"Pearson's correlation: ")
print(pearson_matrix["Step"].sort_values(ascending=False))
print(f" ")


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
print(f" ")



#Logistic Regression
logistic_reg = LogisticRegression(random_state=42, multi_class='ovr')
param_grid_lg = {}  
grid_search_lg = GridSearchCV(logistic_reg, param_grid_lg, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_lg.fit(A_train, B_train)
best_model_lg = grid_search_lg.best_estimator_
print("Best Logistic Regression Model:", best_model_lg)

#Training and testing error for Logistic Regression
B_train_pred_lg = best_model_lg.predict(A_train)
B_test_pred_lg = best_model_lg.predict(A_test)
mae_train_lg = mean_absolute_error(B_train, B_train_pred_lg)
mae_test_lg = mean_absolute_error(B_test, B_test_pred_lg)
print(f"Logistic Regression - MAE (Train): {mae_train_lg}, MAE (Test): {mae_test_lg}")
print(f" ")


#Support vector machine (Linear, polynomial, rbf, and sigmoid)
svc = SVC()
param_grid_svc = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'degree': [0,1,2,3,4,5],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_svc.fit(A_train, B_train)
best_model_svc = grid_search_svc.best_estimator_
print("Best SVM Model GS:", best_model_svc)
random_search_svc = RandomizedSearchCV(svc, param_grid_svc, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
random_search_svc.fit(A_train, B_train)
best_model_svc_rand = random_search_svc.best_estimator_
print("Best SVM Model RS:", best_model_svc_rand)

#Training and Testing error for SVM
B_train_pred_svc = best_model_svc.predict(A_train)
B_test_pred_svc = best_model_svc.predict(A_test)
mae_train_svc = mean_absolute_error(B_train, B_train_pred_svc)
mae_test_svc = mean_absolute_error(B_test, B_test_pred_svc)
print(f"SVM - MAE (Train): {mae_train_svc}, MAE (Test): {mae_test_svc}")
print(f" ")


#Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search_dt.fit(A_train, B_train)
best_model_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)

#Training and testing error for Decision Tree
B_train_pred_dt = best_model_dt.predict(A_train)
B_test_pred_dt = best_model_dt.predict(A_test)
mae_train_dt = mean_absolute_error(B_train, B_train_pred_dt)
mae_test_dt = mean_absolute_error(B_test, B_test_pred_dt)
print(f"Decision Tree - MAE (Train): {mae_train_dt}, MAE (Test): {mae_test_dt}")
print(f" ")












