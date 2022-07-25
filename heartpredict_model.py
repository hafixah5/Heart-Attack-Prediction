# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 10:20:27 2022

@author: fizah
"""
from sklearn.metrics import classification_report,ConfusionMatrixDisplay,confusion_matrix
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os


#%% Constants

pd.set_option('display.max_columns', None)

def cramers_corrected_stat(matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(matrix)[0]
    n = matrix.sum()
    phi2 = chi2/n
    r,k = matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


CSV_PATH = os.path.join(os.getcwd(),'dataset','heart.csv')

BEST_ESTIMATOR_SAVE_PATH = os.path.join(os.getcwd(), 'pickle files','best_estimator.pkl')
#%%

df = pd.read_csv(CSV_PATH)

#%% Data Inspection/Visualization

df.head()
df.info()
df.describe().T

columns = list(df.columns)

cat = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']

cont = ['age','trtbps','chol','thalachh','oldpeak']

#visualization

df.boxplot() 
# Prominent outliers in columns trtbps and chol

# for i in cont:
#     plt.figure()
#     sns.histplot(df[i])
#     plt.show()
    
# for i in cat:
#     plt.figure()
#     sns.countplot(df[i])
#     plt.show()
    
df.groupby(['thall', 'output']).agg({'output':'count'}).plot(kind='bar')
# df.groupby(['thall', 'output']).agg({'thalachh':'count'}).plot(kind='bar')

# Dataset taken from adult population (aged 29 to 77 years old):
# chol range level of 94 - 564 is still accepted in adult (no clipping done)
# resting blood pressure(trtbps) range (94-200) is accepted in adult (no clipping done)

#%% Data Cleaning

df.isna().sum() # no null values
df.duplicated().sum() # 1 duplicate found
df = df.drop_duplicates() #removing the duplicate

print((df['thall'] == 0).sum()) # has two rows with 0 value (null values)
df['thall'].mode()
df['thall'] = df['thall'].replace(0,2)
# print(df.loc[df['chol'] > 500])
# df = df.drop(85)

# No further imputation is done as they data has no null values, at this point.

#%% Feature Selection

df_cont = df.loc[:, ['age','trtbps','chol','thalachh','oldpeak']]
df_cont.corr()
plt.figure()
sns.heatmap(df_cont.corr(), cmap=plt.cm.Reds, annot=True)
plt.show()

# cont vs cat
selected_features = []
for i in cont:
    lr=LogisticRegression()
    y_reg = np.expand_dims(df['output'], axis = -1)
    x_reg = np.expand_dims(df[i],axis =-1)
    lr.fit(x_reg, y_reg)
    print(i, ':', lr.score(x_reg, y_reg))
    if lr.score(x_reg, y_reg) > 0.5:
        selected_features.append(i)

# cat vs cat
for i in cat:
    print(i)  
    matrix = pd.crosstab(df[i], df['output']).to_numpy()
    print(cramers_corrected_stat(matrix))
    if cramers_corrected_stat(matrix) > 0.5:
        selected_features.append(i)

# removing 'output' from selected_features list, as 'output' is the target
selected_features.pop()

# Finalized selected features are:
selected_features = ['age','cp','trtbps','chol','thalachh','oldpeak','thall']

#%% Data Preprocessing

X = df.loc[:, ['age','trtbps','chol','thalachh','oldpeak','cp','thall']]
y = df['output']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,
                                                  random_state=123)


#Pipelines for Decision Tree (tree)
pipeline_mms_tree = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('Tree_Classifier', DecisionTreeClassifier())
    ]) 

pipeline_ss_tree = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('Tree_Classifier', DecisionTreeClassifier())
    ])

#Pipelines([steps]) - LogReg (lr)
pipeline_mms_lr = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('Logistic_Classifier', LogisticRegression())
    ]) 

pipeline_ss_lr = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('Logistic_Classifier', LogisticRegression())
    ])

#Pipeline RandomForest (forest)
pipeline_mms_forest = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('Tree_Classifier', RandomForestClassifier())
    ]) 

pipeline_ss_forest = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('Tree_Classifier', RandomForestClassifier())
    ])

#Pipeline([steps]) - SVM (svm)
pipeline_mms_svm = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('SVC', SVC())
    ]) 

pipeline_ss_svm = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('SVC', SVC())
    ])

#Pipelines for Gradientboost (gb)
pipeline_mms_gb = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('Gradient_Boost', GradientBoostingClassifier())
    ]) 

pipeline_ss_gb = Pipeline([
    ('Standard_Scaler', StandardScaler()),
    ('Gradient_Boost', GradientBoostingClassifier())
    ])

pipelines = [pipeline_mms_tree,pipeline_ss_tree,
             pipeline_mms_lr, pipeline_ss_lr,
             pipeline_mms_forest, pipeline_ss_forest,
             pipeline_mms_svm, pipeline_ss_svm,
             pipeline_mms_gb, pipeline_ss_gb]

for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
best_accuracy = 0

for i,pipe in enumerate(pipelines):
    print(pipe.score(X_test, y_test))
    if pipe.score(X_test, y_test) > best_accuracy:
        best_accuracy = pipe.score(X_test, y_test)
        best_pipeline = pipe
        
print('The best scaler and classifier for Heart Attack Analysis data is {} with accuracy {}'.
      format(best_pipeline.steps, best_accuracy))

# The highest accuracy is achieved by model Logistic Regression with Min-Max Scaling
# The accuracy is 80% 

#%% Tuning

pipeline_mms_lr = Pipeline([
    ('Min_Max_Scaler', MinMaxScaler()),
    ('Logistic_Classifier', LogisticRegression(random_state=123))
    ]) 

grid_param = [{'Logistic_Classifier__max_iter':[100, 1000, 10000],
                'Logistic_Classifier__solver': ['liblinear','lbfgs']}]

grid_search = GridSearchCV(pipeline_mms_lr,param_grid=grid_param,cv=5,
                           verbose=1,n_jobs=-1)

model = grid_search.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.best_index_)
print(model.best_params_)

# Accuracy: 80%
# best params: max_iter (100), solver ('lbfgs')

# Saving the model
with open(BEST_ESTIMATOR_SAVE_PATH, 'wb') as file:
    pickle.dump(model.best_estimator_,file)
    
#classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm=confusion_matrix(y_test,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# True positive value predicte dis 39, and True negative is 34
#%% Test Data

data = {'age': [65,61,45,40,48,41,36,45,57,69],
        'trtbps': [142,140,128,125,132,108,121,111,155,179],
        'chol': [220,207,204,307,254,165,214,198,271,273],
        'thalachh': [158,138,172,162,180,115,168,176,112,151],
        'oldpeak': [2.3,1.9,1.4,0,0,2,0,0,0.8,1.6],
        'cp': [3,0,1,1,2,0,2,0,0,2],
        'thall': [1,3,2,2,2,3,2,2,3,3]}

new_X = pd.DataFrame(data)

y_pred_new = model.predict(new_X)
print(y_pred_new)
# True ouput: [1 0 1 1 1 0 1 0 0 0]
# Output:     [1 0 1 1 1 0 1 1 0 0]
# This model accurately predicted 9 out of 10 values.
# Accuracy is 90%

print(model.score(new_X,y_pred_new))

new_prob = model.predict_proba(new_X)
print(new_prob)

# Individual test data
pred_xx = np.expand_dims([65,142,220,158,2.3,3,1],axis=0)

print(model.predict(pred_xx))

# This also produced the same result as new_X after predicting.









