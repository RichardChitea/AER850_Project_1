import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import StackingClassifier
import joblib
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



#Stratified Sampling
df["Z_cat"] = pd.cut(df['Z'], bins = [0,2,4,6,np.inf], labels = [1,2,3,4])

my_splitter = StratifiedShuffleSplit(n_splits=1,
                                     test_size=0.2,
                                     random_state=42)

for train_index, test_index in my_splitter.split(df,df["Z_cat"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True) 
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
grid_search_lg = GridSearchCV(logistic_reg, param_grid_lg, cv=5, scoring='f1_micro', n_jobs=-1)
grid_search_lg.fit(A_train, B_train)
best_model_lg = grid_search_lg.best_estimator_
print("Best Logistic Regression Model:", best_model_lg)

B_test_lg_gs = grid_search_lg.predict(A_test)
print(classification_report(B_test, B_test_lg_gs))

print(f" ")


#Support vector machine
svc = SVC()
param_grid_svc = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'degree': [0,1,2,3,4,5],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='f1_micro', n_jobs=-1)
grid_search_svc.fit(A_train, B_train)
best_model_svc = grid_search_svc.best_estimator_
print("Best SVM Model GS:", best_model_svc)
random_search_svc = RandomizedSearchCV(svc, param_grid_svc, cv=5, scoring='f1_micro', n_jobs=-1)
random_search_svc.fit(A_train, B_train)
best_model_svc_rand = random_search_svc.best_estimator_
print("Best SVM Model RS:", best_model_svc_rand)

B_test_svc_gs = grid_search_svc.predict(A_test)
print(classification_report(B_test, B_test_svc_gs))

B_test_svc_rs = grid_search_svc.predict(A_test)
print(classification_report(B_test, B_test_svc_rs))

cnf_mat_svc_rs = confusion_matrix(B_test, B_test_svc_rs)
class_names = [1,2,3,4,5,6,7,8,9,10,11,12,13]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(np.abs(cnf_mat_svc_rs), annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Test')
plt.xlabel('Predicted')

print(f" ")


#Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
param_grid_dt = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(decision_tree, param_grid_dt, cv=5, scoring='f1_micro', n_jobs=-1)
grid_search_dt.fit(A_train, B_train)
best_model_dt = grid_search_dt.best_estimator_
print("Best Decision Tree Model:", best_model_dt)

B_test_dt_gs = grid_search_dt.predict(A_test)
print(classification_report(B_test, B_test_dt_gs))

print(f" ")


#Stacked Classifier
estimators = []
estimators.append(('SVM', SVC()))
estimators.append(('Decision Tree', DecisionTreeClassifier(random_state=42)))
stack_class = StackingClassifier(estimators=estimators, final_estimator=logistic_reg, cv=5)
stack_class.fit(A_train, B_train)

B_test_stack_class = stack_class.predict(A_test)
print(classification_report(B_test, B_test_stack_class))

cnf_mat_stack_class = confusion_matrix(B_test, B_test_stack_class)
class_names = [1,2,3,4,5,6,7,8,9,10,11,12,13]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

sns.heatmap(np.abs(cnf_mat_stack_class), annot = True, cmap = "PuBuGn", fmt = 'g')
plt.ylabel('Test')
plt.xlabel('Predicted')


#joblib
pipe = grid_search_svc
joblib.dump(pipe, 'svm.joblib')

dataset = pd.read_csv("dataset.csv")

dataset_pred = pipe.predict(dataset)
print(dataset_pred)
