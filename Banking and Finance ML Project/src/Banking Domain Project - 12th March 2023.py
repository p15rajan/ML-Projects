#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic requirement
import os, sys
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

import sklearn


# ![image.png](attachment:image.png)

# In[2]:


geo = pd.read_csv('Geo_scores.csv')
instance = pd.read_csv('instance_scores.csv')
lambda_wts = pd.read_csv('Lambda_wts.csv')
qset = pd.read_csv('Qset_tats.csv')
test = pd.read_csv('test_share.csv')
train = pd.read_csv('train.csv')


# In[3]:


print(geo.head(2))
print("****************"*5)
print(instance.head(2))
print("****************"*5)
print(lambda_wts.head(2))
print("****************"*5)
print(qset.head(2))
print("****************"*5)
print(test.head(2))
print("****************"*5)
print(train.head(2))
print("****************"*5)


# In[4]:


print(geo.isnull().sum())
print("****************"*5)
print(instance.isnull().sum())
print("****************"*5)
print(lambda_wts.isnull().sum())
print("****************"*5)
print(qset.isnull().sum())
print("****************"*5)
print(test.isnull().sum())
print("****************"*5)
print(train.isnull().sum())
print("****************"*5)


# In[5]:


print(geo.isnull().sum()/len(geo)*100)
print("****************"*5)
print(qset.isnull().sum()/len(qset)*100)


# In[6]:


print(geo.shape)
print("****************"*5)
print(instance.shape)
print("****************"*5)
print(lambda_wts.shape)
print("****************"*5)
print(qset.shape)
print("****************"*5)
print(test.shape)
print("****************"*5)
print(train.shape)
print("****************"*5)


# In[7]:


print(geo['id'].nunique())
print("****************"*5)
print(instance['id'].nunique())
print("****************"*5)
print(lambda_wts['Group'].nunique())
print("****************"*5)
print(qset['id'].nunique())
print("****************"*5)
print("Test", test['id'].nunique())
print("Test", test['Group'].nunique())
print("****************"*5)
print("Train", train['id'].nunique())
print("Train", train['Group'].nunique())
print("****************"*5)


# In[8]:


56962+227845


# In[9]:


1301+915


# In[10]:


print(geo.info())
print("****************"*5)
print(qset.info())


# In[11]:


print(geo.describe())
print("****************"*5)
print(qset.describe())


# In[12]:


sns.boxplot(y='geo_score', data=geo)


# In[13]:


sns.boxplot(y='qsets_normalized_tat', data=qset)


# In[14]:


geo['geo_score'].mean()


# In[15]:


geo['geo_score'].median()


# In[16]:


qset['qsets_normalized_tat'].mean()


# In[17]:


qset['qsets_normalized_tat'].median()


# In[18]:


geo.fillna(0.18, inplace=True)
qset.fillna(0.019, inplace=True)


# In[19]:


print(geo['id'].nunique())
print("****************"*5)
print(instance['id'].nunique())
print("****************"*5)
print(lambda_wts['Group'].nunique())
print("****************"*5)
print(qset['id'].nunique())
print("****************"*5)
print("Test", test['id'].nunique())
print("Test", test['Group'].nunique())
print("****************"*5)
print("Train", train['id'].nunique())
print("Train", train['Group'].nunique())
print("****************"*5)


# In[20]:


geo = geo.groupby('id').mean()
instance = instance.groupby('id').mean()
qset = qset.groupby('id').mean()


# In[21]:


print(geo.shape)
print("****************"*5)
print(instance.shape)
print("****************"*5)
print(lambda_wts.shape)
print("****************"*5)
print(qset.shape)
print("****************"*5)
print(test.shape)
print("****************"*5)
print(train.shape)
print("****************"*5)


# In[22]:


train['data'] = 'train'
test['data'] = 'test'


# In[23]:


train.shape


# In[24]:


test.shape


# In[25]:


print(train.head(1))
print("*************************"*5)
print(test.head(1))


# In[26]:


all_data = pd.concat([train, test], axis=0)


# In[27]:


all_data.shape


# In[28]:


print(lambda_wts['Group'].nunique())
print("****************"*5)
print(all_data['Group'].nunique())


# In[29]:


all_data = pd.merge(all_data, lambda_wts, on='Group', how='left')


# In[30]:


all_data.shape


# In[31]:


all_data.head()


# In[32]:


all_data = pd.merge(all_data, geo, on='id', how='left')


# In[33]:


all_data.shape


# In[34]:


all_data = pd.merge(all_data, instance, on='id', how='left')


# In[35]:


all_data = pd.merge(all_data, qset, on='id', how='left')


# In[36]:


all_data.shape


# In[37]:


all_data.head(2)


# In[38]:


# 


# In[39]:


train = all_data[all_data['data']=='train']
test = all_data[all_data['data']=='test']


# In[40]:


train.shape


# In[41]:


train.columns


# In[42]:


x1 = train.drop(['id', 'Group','Target', 'data'], axis=1)
y1 = train['Target']


# In[43]:


x1.head()


# In[44]:


x1.info()


# In[45]:


# Feature scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler_x = scaler.fit_transform(x1)


# In[46]:


scaler_x


# In[ ]:





# In[47]:


y1.head()


# In[48]:


y1.value_counts()


# In[49]:


outlier_fraction = 394/(227451+394)
outlier_fraction


# In[ ]:





# In[50]:


test.shape


# In[51]:


test.isnull().sum()


# In[52]:


future_prediction_data = test.drop(['id', 'Group','Target', 'data'], axis=1)


# In[53]:


future_prediction_data.head(2)


# In[54]:


future_prediction_data = scaler.fit_transform(future_prediction_data)


# In[55]:


future_prediction_data.shape


# In[ ]:





# # Split the data into training and test

# In[56]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(scaler_x, y1, test_size=0.3, random_state=101)


# In[ ]:





# # Stacking Classifier

# In[57]:


#!pip install mlxtend


# In[58]:


from mlxtend.classifier import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score


# In[ ]:


clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = RandomForestClassifier()
clf4 = KNeighborsClassifier()
clf5 = GaussianNB()
clf6 = SVC()

sclf = StackingClassifier(classifiers=[clf2,clf3,clf4,clf5,clf6], meta_classifier=clf1)

print('5-fold cross validation : \n')

for clf, label in zip([clf2,clf3,clf4,clf5,clf6, sclf], ['DT', 'RF', ' KNN', 'NiaveBayes','SVM','Stacking']):
    scores = cross_val_score(clf, x_train, y_train, cv=5, scoring='accuracy')
    scores_test = cross_val_score(clf, x_test, y_test, cv=5, scoring='accuracy')
    print("Accuracy : %0.2f (+/-%0.2f)[%s]" % (scores.mean(), scores.std(), label))
    print("Test Accuracy : %0.2f (+/-%0.2f)[%s]" % (scores_test.mean(), scores_test.std(), label))


# In[ ]:


sclf.fit(x_train, y_train)
y_pred_train = sclf.preidct(x_train)
y_pred_test = sclf.predict(x_test)


# In[ ]:


print(accuracy_score(y1, y_pred))
print(classification_report(y1, y_pred))
print(confusion_matrix(y1, y_pred))


# In[ ]:


final_pred = sclf.predict(future_prediction_data)


# In[ ]:


# Isolation Forest, Local Outlier Factor and OneClassSVM


# In[59]:


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


# In[ ]:


OneClassSVM()


# In[60]:


classification = {'IsolationForest' : IsolationForest(contamination=outlier_fraction),
                 'LocalOutlierFactor' : LocalOutlierFactor(contamination=outlier_fraction),
                 'OneClassSVM' : OneClassSVM()}


# In[61]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[ ]:


n_outlier = 394

for i , (clf_name, clf) in enumerate(classification.items()):
    if clf_name == 'LocalOutlierFactor' :
        y_pred = clf.fit_predict(scaler_x)
        score_predict = clf.negative_outlier_factor_
    elif clf_name =='OneClassSVM' :
        clf.fit(scaler_x)
        y_pred = clf.predict(scaler_x)
    else:
        clf.fit(scaler_x)
        score_prediction = clf.decision_function(scaler_x)
        y_pred = clf.predict(scaler_x)
        
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    n_error = (y_pred !=y1).sum()
    
    print("{} : {}".format(clf_name, n_error))
    print(accuracy_score(y1, y_pred))
    print(classification_report(y1, y_pred))
    print(confusion_matrix(y1, y_pred))
        

