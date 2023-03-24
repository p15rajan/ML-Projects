#!/usr/bin/env python
# Make necessary import 
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set()
import warnings
warnings.filterwarnings('ignore')
# Load the file
df = pd.read_csv('/content/kidney_disease.csv')
df.head(10)
df.info()

df.isnull().sum()/len(df)*100

# Imputing null values
from sklearn.impute import SimpleImputer
imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df_imputed = pd.DataFrame(imp_mode.fit_transform(df))
df_imputed.columns = df.columns
df_imputed
df_imputed.isnull().sum()/len(df)*100

for i in df_imputed.columns:
  print("*********************************************************",
        i , 
        "************************************************************")
  print()
  print(set(df_imputed[i].tolist()))
  print()

df_imputed['rc'].mode()

df_imputed['rc'] = df_imputed['rc'].apply(lambda x:'5.2' if x=='\t?' else x)
df_imputed['bp'] = np.where(df_imputed['bp']==0, df_imputed['bp'].median(), df_imputed['bp'])
df_imputed['hemo'] = np.where(df_imputed['hemo']==0, df_imputed['hemo'].median(), df_imputed['hemo'])


# In[11]:


df_imputed['pcv'].mode()
df_imputed['pcv'] = df_imputed['pcv'].apply(lambda x:'43' if x=='\t43' else x)
df_imputed['pcv'] = df_imputed['pcv'].apply(lambda x:'41' if x=='\t?' else x)


# In[12]:


df_imputed['wc'].mode()


# In[13]:


df_imputed['wc'] = df_imputed['wc'].apply(lambda x:'9800' if x=='\t?' else x)
df_imputed['wc'] = df_imputed['wc'].apply(lambda x:'6200' if x=='\t6200' else x)
df_imputed['wc'] = df_imputed['wc'].apply(lambda x:'8400' if x=='\t8400' else x)


df_imputed['cad'] = df_imputed['cad'].apply(lambda x:'no' if x=='\tno' else x)
df_imputed['classification'] = df_imputed['classification'].apply(lambda x:'ckd' if x=='ckd\t' else x)

df_imputed['dm'] = df_imputed['dm'].apply(lambda x:'yes' if x=='\tyes' else x)
df_imputed['dm'] = df_imputed['dm'].apply(lambda x:'no' if x=='\tno' else x)
df_imputed['dm'] = df_imputed['dm'].apply(lambda x:'yes' if x==' yes' else x)

df_imputed.info()


for i in df.select_dtypes(exclude=['object']).columns:
  df_imputed[i] = df_imputed[i].apply(lambda x: float(x))

df_imputed.info()


df_imputed['pcv'] = pd.to_numeric(df_imputed['pcv'])
df_imputed['wc'] = pd.to_numeric(df_imputed['wc'])
df_imputed['rc'] = pd.to_numeric(df_imputed['rc'])

df_imputed.head()

def boxplots(col):
  sns.boxplot(df_imputed[col])
  plt.show()

for i in list(df_imputed.select_dtypes(exclude=['object']).columns)[1:]:
  boxplots(i)
"""
def distplots(col):
  sns.distplot(df[col])
  plt.show()

for i in list(df_imputed.select_dtypes(exclude=['object']).columns)[1:]:
  distplots(i)"""

df_imputed['classification'].value_counts()


# split the data into x and y
x = df_imputed.iloc[:,:-1]
y = df_imputed.iloc[:,-1]

x.head(1)

x.info()

x.columns

# one hot encoder and dummy variable
x = pd.get_dummies(x, columns=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], drop_first=True)

x.head()

x = x.drop(['id'], axis=1)
x.head()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaler = scaler.fit_transform(x)

y.head(1)


# PCA method for understanding only
x.shape

from sklearn.decomposition import PCA
pca = PCA(0.95)
X_PCA = pca.fit_transform(x_scaler)
print(x.shape)
print(X_PCA.shape)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Just for reference
lda = LDA()
X_LDA = lda.fit_transform(x_scaler, y)
print(x.shape)
print(X_LDA.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.2, random_state=1)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.naive_bayes import BernoulliNB

list_met = []
list_accuracy = []

# Logistic Regression
logit = LogisticRegression()
logit = logit.fit(x_train, y_train)
y_pred_lr = logit.predict(x_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)

# DecisionTree
dt = DecisionTreeClassifier()
dt = dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# RandomForestClassifier
rf = RandomForestClassifier()
rf = rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# SVC
svc = SVC()
svc = svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)

# KNeighborsClassifier
knn = KNeighborsClassifier()
knn = knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# GaussianNB
nb = GaussianNB()
nb = nb.fit(x_train, y_train)
y_pred_nb = nb.predict(x_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Combining all the above model with voting classifier
model_evc = VotingClassifier(estimators=[('logit', logit),('dt',dt),('rf', rf),('svc', svc),('knn', knn),('nb', nb)], voting='hard')
model_evc = model_evc.fit(x_train, y_train)
pred_evc = model_evc.predict(x_test)
accuracy_evc = accuracy_score(y_test, pred_evc)

list1 = ['LogisticRegression','DecisionTree','RandomForest','Support Vector Machine','K Nearest Neighbors','Naive Bayes Theorem','Voting']

list2 = [accuracy_lr, accuracy_dt, accuracy_rf, accuracy_svc, accuracy_knn, accuracy_nb, accuracy_evc]

list3 = [logit, dt, rf, svc, knn, nb, model_evc]

df_accuracy = pd.DataFrame({'Method Used': list1, "Accuracy": list2})

print(df_accuracy)

chart = sns.barplot(x = 'Method Used', y='Accuracy', data=df_accuracy)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
print(chart)

pred_evc_train = model_evc.predict(x_train)
pred_evc_test = model_evc.predict(x_test)

accuracy_evc_training = accuracy_score(y_train, pred_evc_train)
accuracy_evc_test = accuracy_score(y_test, pred_evc_test)

print(accuracy_evc_training)
print("***********************************************")
print(accuracy_evc_test)


y_pred_rf_train = rf.predict(x_train)
y_pred_rf_test = rf.predict(x_test)

accuracy_rf_training = accuracy_score(y_train, y_pred_rf_train)
accuracy_rf_test = accuracy_score(y_test, y_pred_rf_test)

print(accuracy_rf_training)
print("***********************************************")
print(accuracy_rf_test)

classification_rf_training = classification_report(y_train, y_pred_rf_train)
classification_rf_test = classification_report(y_test, y_pred_rf_test)

print(classification_rf_training)
print("***********************************************")
print(classification_rf_test)



# # HyperParameter Tuning

# Grid Search CV

from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'n_estimators':[7,9,12,15,16,20,17,18,25,30], 'max_depth':[2,3,4,5,None],
                     'class_weight':[None, {0:0.33, 1:0.67}, 'balanced'], 'random_state':[42]}]
clf = GridSearchCV(RandomForestClassifier(criterion='entropy'), tuned_parameters, cv=10)
clf.fit(x_train, y_train)

print("Detailed Classification Report :")
y_true, lr_pred = y_test, clf.predict(x_test)
print(classification_report(y_true, lr_pred))   
print("************************************")
print(accuracy_score(y_true, lr_pred))                  

GridSearchCV()

