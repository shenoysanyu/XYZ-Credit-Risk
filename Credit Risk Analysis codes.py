#!/usr/bin/env python
# coding: utf-8

# # Credit Risk Analysis

# # *Importing Libraries*

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


##### Importing Dataset from Text File


Credit_DS = pd.read_csv(r"C:\Users\Sanyu\Imarticus\Projects\Python Project\Credit_Risk_Analysis.txt", delimiter="\t", header=0, engine='python')
print(Credit_DS)
#%%
Credit_DS.shape

Credit_DS.head()

print(Credit_DS.dtypes)

Credit_DS.describe()
 
### Creating A Copy of DataFrame

Credit_DS_Backup = Credit_DS.copy()

Credit_DS_Backup.shape

print(Credit_DS_Backup)
#%%
##### Pre Processing the Data

### Find Missing Value and Calculating Missing Value Percentage. 
#%%
Count_Null=Credit_DS.isnull().sum()
print(Count_Null)
#%%
Per_Null =round(Credit_DS.isnull().sum()/len(Credit_DS) * 100,2)
print(Per_Null)
#%%
Missing_Val= pd.DataFrame({'count':Count_Null,'% Count':Per_Null})
print(Missing_Val)
#%%
Missing_Val.sort_values(by=['% Count'],ascending=False)
#%%
Missing_Val.to_csv(r'D:\Desktop\MissingValues.csv')
#%%
### Deleting Variable having Missing Values More than 50%
#%%
Credit_DS = Credit_DS.loc[:,Credit_DS.isnull().sum()/len(Credit_DS) <.50 ]
#%%
Credit_DS.shape
#%%
Credit_DS.to_csv(r'D:\Desktop\Rem_Data.csv')
#%%
### Deleting Variables based on Domain Knowledge

Credit_DS=Credit_DS.drop(['id','member_id','sub_grade','emp_title','title','zip_code','addr_state','earliest_cr_line','inq_last_6mths',
                          'last_pymnt_d','next_pymnt_d','last_credit_pull_d','pymnt_plan','policy_code'],axis=1)

#%%
Credit_DS.shape
#%%
Credit_DS.to_csv(r'D:\Desktop\Rem_FinalData.csv')
#%%
Credit_DS.isnull().sum()
#%%
##### Assigning Label to Categorical Data and Converting Numerical 

### Term Column Labelling

#%%
Credit_DS.term.value_counts()
#%%
Credit_DS.term.value_counts()
Credit_DS.term=Credit_DS.term.str.extract('(\d+)')
term_final={'36':0,'60':1}
Credit_DS.term=[term_final[item]for item in Credit_DS.term]
print(Credit_DS.term)

# 36 Months - 0
# 60 Months - 1
#%%
### Grade Column Labelling

Credit_DS.grade.value_counts()
#%%
Credit_DS.grade.value_counts()
grade_final={'A':6,'B':5,'C':4,'D':3,'E':2,'F':1,'G':0}
Credit_DS.grade=[grade_final[item]for item in Credit_DS.grade]
print(Credit_DS.grade)
#%%

### Emp_Length Columns Labeling

A = Credit_DS.emp_length.isnull().sum()
A
#%%
Credit_DS.emp_length.value_counts()
#%%
Credit_DS['emp_length'].mode()[0]
Credit_DS['emp_length'].fillna(Credit_DS['emp_length'].mode()[0],inplace=True)
Credit_DS['emp_length'] = Credit_DS['emp_length'].str.replace('+','')
Credit_DS['emp_length'] = Credit_DS['emp_length'].str.replace('<','')
print(Credit_DS.emp_length)
Credit_DS.emp_length=Credit_DS.emp_length.str.extract('(\d+)')
print(Credit_DS.emp_length)
Credit_DS.emp_length = [int(x) for x in Credit_DS.emp_length]
print(Credit_DS.emp_length.dtype)
#%%
###Home_Ownership labeling
#%%
Credit_DS.home_ownership.value_counts()
#%%
print(Credit_DS['home_ownership'].unique())
home_ownership_score={'NONE':0,'ANY':0,'OTHER':0,'RENT':1,'MORTGAGE':2,'OWN':3}
Credit_DS.home_ownership=[home_ownership_score[item]for item in Credit_DS.home_ownership]
print(Credit_DS.home_ownership)
#%%

### Verification_statusColumns Labeling
#%%
Credit_DS.verification_status.value_counts()
#%%
print(Credit_DS['verification_status'].unique())
Ver_status={'Verified':1,'Source Verified':1,'Not Verified':0}
Credit_DS.verification_status=[Ver_status[item]for item in Credit_DS.verification_status]
print(Credit_DS.verification_status)
#%%
### Purpose Plan Columns Labeling
#%%
Credit_DS.purpose.value_counts()
#%%
print(Credit_DS['purpose'].unique())
purpose_score={'vacation':0,'other':1,'major_purchase':2,'car':3,'wedding':4,'medical':5,'moving':6,'home_improvement':7,'credit_card':8,'debt_consolidation':9,'house':10,'small_business':11,'renewable_energy':12,'educational':13}
Credit_DS.purpose=[purpose_score[item] for item in Credit_DS.purpose]
print(Credit_DS.purpose)
#%%

### Treament of Missing value of Revol_util Columns 

Credit_DS.revol_util.describe()
print(Credit_DS.revol_util.isnull().sum())
Credit_DS['revol_util'].fillna(Credit_DS['revol_util'].mean(),inplace=True)
print(Credit_DS.revol_util.isnull().sum())
#%%
### Initial List Status Columns Labeling
#%%
Credit_DS.initial_list_status.value_counts()
#%%
print(Credit_DS['initial_list_status'].unique())
list_status={'w':1,'f':2}
Credit_DS.initial_list_status=[list_status[item] for item in Credit_DS.initial_list_status]
print(Credit_DS.initial_list_status)
#%%

### ApplicationType Columns Labeling
#%%
Credit_DS.application_type.value_counts()
#%%
print(Credit_DS['application_type'].unique())
App_type={'INDIVIDUAL':1,'JOINT':2}
Credit_DS.application_type=[App_type[item] for item in Credit_DS.application_type]
print(Credit_DS.application_type)
#%%

### Open Account Labeling
#%%
Credit_DS.open_acc.value_counts()
#%%
Credit_DS.open_acc.mode()

def open_acc_final(i):
    if i in range (4,25):
        return(1)
    else:
        return(0)


Credit_DS['open_acc']= Credit_DS['open_acc'].apply(open_acc_final)
print(Credit_DS.open_acc) 
#%%

### Treating Missing Value for collections_12_mths_ex_med 
Credit_DS.collections_12_mths_ex_med.value_counts()
#%%
Credit_DS.collections_12_mths_ex_med.describe()
print(Credit_DS.collections_12_mths_ex_med.isnull().sum())
Credit_DS['collections_12_mths_ex_med'].fillna(Credit_DS['collections_12_mths_ex_med'].median(),inplace=True)
print(Credit_DS.collections_12_mths_ex_med.isnull().sum())
#%%

### Treating Missing Value for tot_coll_amt
#%%
Credit_DS.tot_coll_amt.describe()
print(Credit_DS.tot_coll_amt.isnull().sum())
Credit_DS['tot_coll_amt'].fillna(Credit_DS['tot_coll_amt'].mean(),inplace=True)
print(Credit_DS.tot_coll_amt.isnull().sum())
#%%

### Treating Missing Value for tot_cur_bal

Credit_DS.tot_cur_bal.describe()
print(Credit_DS.tot_cur_bal.isnull().sum())
Credit_DS['tot_cur_bal'].fillna(Credit_DS['tot_cur_bal'].mean(),inplace=True)
print(Credit_DS.tot_cur_bal.isnull().sum())
#%%

### Treating Missing Value for total_rev_hi_lim
#%%
Credit_DS.total_rev_hi_lim.describe()
print(Credit_DS.total_rev_hi_lim.isnull().sum())
Credit_DS['total_rev_hi_lim'].fillna(Credit_DS['total_rev_hi_lim'].mean(),inplace=True)
print(Credit_DS.total_rev_hi_lim.isnull().sum())
#%%
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from statistics import mode
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.preprocessing import Imputer, RobustScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             accuracy_score, roc_curve,
                             precision_recall_curve, f1_score)
from sklearn.pipeline import make_pipeline
from scipy.stats import boxcox
from sklearn.metrics import confusion_matrix

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")
sns.set_context("notebook")
#%%
### Heat Map for Multi Colinearity
#%%
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")
sns.set_context("notebook")
plt.figure(figsize=(20,20)) 
sns.set_context("paper", font_scale=1.2) 
#%%
sns.heatmap(Credit_DS.assign(grade=Credit_DS.grade.astype('category').cat.codes, 
                         term=Credit_DS.term.astype('category').cat.codes, 
                         verification=Credit_DS.verification_status.astype('category').cat.codes, 
                         emp_length=Credit_DS.emp_length.astype('category').cat.codes, 
                          home=Credit_DS.home_ownership.astype('category').cat.codes, 
                         purpose=Credit_DS.purpose.astype('category').cat.codes,
                            list_status=Credit_DS.initial_list_status.astype('category').cat.codes,
                            application=Credit_DS.application_type.astype('category').cat.codes,
                            openacc=Credit_DS.open_acc.astype('category').cat.codes).corr(),
             annot=False, cmap='PuBuGn', vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.show()
#%%
### Split Train and Test data with issue_d date.
#%%
Credit_DS.issue_d=pd.to_datetime(Credit_DS.issue_d,infer_datetime_format=True)
col_name='issue_d'
print(Credit_DS[col_name].dtype)
#%%
split_data="2015-06-01"
Training=Credit_DS[Credit_DS['issue_d']<split_data]
Training.shape
#%%
Testing=Credit_DS.loc[Credit_DS['issue_d']>='2015-06-01',:]
Testing.shape
#%%
Training=Training.drop(['issue_d'],axis=1)
Testing=Testing.drop(['issue_d'],axis=1)
#%%
X_train = Training.values[:,:-1]
Y_train = Training.values[:,-1]
#%%
X_test = Testing.values[:,:-1]
Y_test = Testing.values[:,-1]

### Scaling The Data
#%%
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print(X_train)
print(X_test)

#%%
Y_train=Y_train.astype(int)
Y_test=Y_test.astype(int)
#%%

### Logistic Regression MODEL BUILDING
#%%

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression()
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))
#%%

print(classifier.coef_)
print(classifier.intercept_)
#%%
### Model Evaluation
#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)
print("Classification Report :")
print(classification_report(Y_test, Y_pred))
acc= accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)
#%%

fig, axes = plt.subplots(figsize=(8,6))
cm = confusion_matrix(Y_test, Y_pred)
cm = (cm.astype('float')/cm.sum(axis=0))*100
ax=sns.heatmap(cm,annot=True,cmap='magma');
ax.set_xlabel("Actual Values")
ax.set_ylabel("Our Predicted Values")
ax.axis('equal')

#%%
y_pred_prob= classifier.predict_proba(X_test)
print(y_pred_prob)
#%%
##TO FIND WHICH THRESHOLD IS BETTER
for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :",
          cfm[1,0]," , type 1 error:", cfm[0,1])
#%%
from sklearn import metrics
fpr , tpr , z = metrics.roc_curve(Y_test , y_pred_prob[:,1])
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)

#%%
#ROC curve
import matplotlib.pyplot as plt 
#%matplotlib inline 
plt.title('Receiver Operating Characteristic') 
plt.plot(fpr, tpr, 'b', label = auc) 
plt.legend(loc = 'lower right') 
plt.plot([0, 1], [0, 1],'r--') 
plt.xlim([0, 1]) 
plt.ylim([0, 1]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.show()

#%%
##K - FOLD METHOD FOR CROSS VALIDATION
#Using cross validation
classifier=(LogisticRegression())
#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

#%%
from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,
                                    y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())
#Gives multiple accuracies and finally gives mean of all accuracies
#82.40% is approx close to 85% , hence we go ahead with base model
#if we want to go ahead with kfold model then execute the next block
#%%
for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])
#%%
Y_pred=classifier.predict(X_test)
#%%
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)

print("Classification report: ")
#recall gives accuracy for individual value(sensitivity for 0(TNR),specificity for 1(TPR))
#precision gives precision of class 0 and class 1 (how many predictions were correct?)
print(classification_report(Y_test, Y_pred))

acc= accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)
#%%

### Decision Tree Model

#%%
from sklearn.tree import DecisionTreeClassifier
model_DT = DecisionTreeClassifier(random_state=10,min_samples_leaf=100,max_depth=25,criterion='gini')
#default criterion is gini
model_DT.fit(X_train,Y_train)
#%%

Y_pred = model_DT.predict(X_test)
#print(Y_pred)
print(list(zip(Y_test,Y_pred)))
#%%

### Evaluation of Model

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cfm = confusion_matrix(Y_test, Y_pred)
print(cfm)

print("Classification report: ")

print(classification_report(Y_test, Y_pred))

acc= accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)
#%%

fig, axes = plt.subplots(figsize=(8,6))
cm = confusion_matrix(Y_test, Y_pred)
cm = (cm.astype('float')/cm.sum(axis=0))*100
ax=sns.heatmap(cm,annot=True,cmap='magma');
ax.set_xlabel("Actual Values")
ax.set_ylabel("Our Predicted Values")
ax.axis('equal')
#%%

colname=Training.columns[:]
colname

#%%
from sklearn import tree

with open("model_DecisionTree.txt","w") as f:
    f = tree.export_graphviz(model_DT,feature_names=colname[:-1],out_file=f)
#%%

from sklearn import metrics
fpr , tpr , z = metrics.roc_curve(Y_test , Y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)
#%%

#ROC curve
import matplotlib.pyplot as plt 
#%matplotlib inline 
plt.title('Receiver Operating Characteristic') 
plt.plot(fpr, tpr, 'b', label = auc) 
plt.legend(loc = 'lower right') 
plt.plot([0, 1], [0, 1],'r--') 
plt.xlim([0, 1]) 
plt.ylim([0, 1]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.show()
#%%

### ENSEMBLE MODEL

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

#%%

estimators = []
model1 = LogisticRegression()
estimators.append(('log', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC(kernel='rbf',gamma=0.1,C=70.0)
estimators.append(('svm', model3))
print(estimators)

#%%

ensemble = VotingClassifier(estimators)
ensemble.fit(X_train,Y_train)
Y_pred=ensemble.predict(X_test)
#%%

### Gradient Boosting Classifier
#%%
from sklearn.ensemble import GradientBoostingClassifier

model_GradientBoosting=GradientBoostingClassifier()
#model_GradientBoosting=DecisionTreeClassifier()

#fit the model on the data and predict the values
model_GradientBoosting.fit(X_train,Y_train)

Y_pred=model_GradientBoosting.predict(X_test)

#checking result
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion_matrix
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
#classification_report
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
#accuracy_score
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%%

fig, axes = plt.subplots(figsize=(8,6))
cm = confusion_matrix(Y_test, Y_pred)
cm = cm.astype('float')/cm.sum(axis=0)
ax=sns.heatmap(cm,annot=True,cmap='magma');
ax.set_xlabel("Actual Values")
ax.set_ylabel("Our Predicted Values")
ax.axis('equal')

#%%

from sklearn import metrics
fpr , tpr , z = metrics.roc_curve(Y_test , Y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)

#%%

#ROC curve
import matplotlib.pyplot as plt 
#%matplotlib inline 
plt.title('Receiver Operating Characteristic') 
plt.plot(fpr, tpr, 'b', label = auc) 
plt.legend(loc = 'lower right') 
plt.plot([0, 1], [0, 1],'r--') 
plt.xlim([0, 1]) 
plt.ylim([0, 1]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.show()
#%%

### ADA Booster Classifier
#%%
from sklearn.ensemble import AdaBoostClassifier
ada_model=AdaBoostClassifier()
ada_model.fit(X_train,Y_train)
Y_pred=ada_model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%%

fig, axes = plt.subplots(figsize=(8,6))
cm = confusion_matrix(Y_test, Y_pred)
cm = (cm.astype('float')/cm.sum(axis=0))*100
ax=sns.heatmap(cm,annot=True,cmap='magma');
ax.set_xlabel("Actual Values")
ax.set_ylabel("Our Predicted Values")
ax.axis('equal')

#%%
from sklearn import metrics
fpr , tpr , z = metrics.roc_curve(Y_test , Y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)

#%%

#ROC curve
import matplotlib.pyplot as plt 
#%matplotlib inline 
plt.title('Receiver Operating Characteristic') 
plt.plot(fpr, tpr, 'b', label = auc) 
plt.legend(loc = 'lower right') 
plt.plot([0, 1], [0, 1],'r--') 
plt.xlim([0, 1]) 
plt.ylim([0, 1]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.show()
#%%

### Extra Tree Classifier
#%%
from sklearn.ensemble import ExtraTreesClassifier
model=(ExtraTreesClassifier(4,random_state=100))
model=model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm)
print("Classification report: ")
print(classification_report(Y_test,Y_pred))
acc=accuracy_score(Y_test, Y_pred)
print("Accuracy of the model: ",acc)

#%%

fig, axes = plt.subplots(figsize=(8,6))
cm = confusion_matrix(Y_test, Y_pred)
cm = (cm.astype('float')/cm.sum(axis=0))*100
ax=sns.heatmap(cm,annot=True,cmap='magma');
ax.set_xlabel("Actual Values")
ax.set_ylabel("Our Predicted Values")
ax.axis('equal')
#%%

from sklearn import metrics
fpr , tpr , z = metrics.roc_curve(Y_test , Y_pred)
auc = metrics.auc(fpr,tpr)
print(auc)
print(fpr)
print(tpr)
#%%

#ROC curve
import matplotlib.pyplot as plt 
#%matplotlib inline 
plt.title('Receiver Operating Characteristic') 
plt.plot(fpr, tpr, 'b', label = auc) 
plt.legend(loc = 'lower right') 
plt.plot([0, 1], [0, 1],'r--') 
plt.xlim([0, 1]) 
plt.ylim([0, 1]) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate')
plt.show()
#%%




