#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import os, sys
import warnings 
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("airline_passenger_satisfaction.csv")


# In[3]:


df.info()


# In[4]:


df['Customer Type'].value_counts()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.columns


# In[8]:


df['Arrival Delay'] = df['Arrival Delay'].fillna(df['Arrival Delay'].median())


# In[9]:


from dataprep.eda import create_report
create_report(df).show()


# In[10]:


df.isnull().sum()


# In[11]:


df.head()


# In[12]:


df.duplicated().sum()


# In[13]:


df['Satisfaction'].value_counts() # Data is balanced 


# In[14]:


df.columns


# In[15]:


df1 = df.drop(['ID','Satisfaction'], axis=1)


# In[16]:


df1.info()


# In[17]:


df_new = df.drop(['ID','Gate Location', 'Leg Room Service', ], axis=1)


# In[18]:


df_new.head()


# In[19]:


# Data Visuzalzation 


# In[20]:


df_new.info()


# In[21]:


# Encoding the Object variables
   
df_new = pd.get_dummies(df_new, columns=['Customer Type','Class', 'Gender', 'Type of Travel'])  


# In[22]:


df_new.head()


# In[23]:


# Defining Independent and Dependent variables 
X = df_new.drop(['Satisfaction'], axis=1)


# In[24]:


X.shape


# In[25]:


Y=df_new['Satisfaction']
arr1 = np.array(Y)
arr2 = arr1.reshape(-1,1)
arr2


# In[26]:


Y = pd.DataFrame(arr2,).reset_index
print(Y)


# In[27]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
Y = encoder.fit_transform(arr2)


# In[28]:


print(Y)


# In[29]:


df_new['Class_Economy Plus'].value_counts()


# In[30]:


for i in X:
    plt.figure()
    plt.tight_layout()
    sns.set(rc={"figure.figsize":(3, 3)})
    f, (ax_box) = plt.subplots(1, sharex=True)
    f, (ax_hist) = plt.subplots(1, sharex=True)
    plt.gca().set(xlabel=i,ylabel='Satisfaction')
    sns.boxplot(X[i], hue=Y,ax=ax_box , linewidth= 1.0)
    sns.histplot(X[i], ax=ax_hist , bins = 10,kde=True) 


# In[31]:


from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()


# In[32]:


X_Scaled = Scaler.fit_transform(X)


# In[33]:


from sklearn.model_selection import train_test_split


# In[34]:


X_train, X_test,Y_train,Y_test = train_test_split(X_Scaled,Y, test_size=0.2, random_state=42)


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,f1_score


# In[36]:


Model_comparison = {}


# ### Logistic Regression 

# In[40]:


print("\033[5m"+ "LOGISTIC REGRESSION" + "\033[0m\n")
Log = LogisticRegression()
Log.fit(X_train,Y_train)


# ### Random Forest Classifier 

# In[41]:


print("\033[5m"+ "RANDOM FOREST" +"\033[0m\n")
RF = RandomForestClassifier()
RF.fit(X_train,Y_train)


# ### Decision Tree Classifier

# In[42]:


print("\033[5m"+ "DECISION TREE CLASSIFIER"+ "\033[0m\n")
DT = DecisionTreeClassifier()
DT.fit(X_train,Y_train)


# ### Naive Bayes Theorem

# In[43]:


print("\033[5m"+ "NAIVE BAYES CLASSIFIER"+"\033[0m\n")
NB = GaussianNB()
NB.fit(X_train,Y_train)


# ### Support Vector Machine

# In[44]:


print("\033[5m"+ "SUPPORT VECTOR MACHINE"+"\033[0m\n")
SVM = SVC()
SVM.fit(X_train,Y_train)


# ### XGboost Classifier 

# In[45]:


print("\033[5m"+ "XGBOOST CLASSIFIER"+"\033[0m\n")
Xg =  XGBClassifier()
Xg.fit(X_train,Y_train)


# ### Stochastic gradient descient (SGD)

# In[46]:


#from sklearn.calibration import CalibratedClassifierCV
#calibrator = CalibratedClassifierCV(ClfSGD, cv='prefit')
#calibrator.fit(X_train,Y_train)


# In[47]:


print("\033[5m"+ "Stochastic Boost Classifier"+"\033[0m\n")
SGD =  SGDClassifier(loss='hinge', alpha=0.0001, class_weight='balanced')
SGD.fit(X_train,Y_train)


# ### Adaboost Classifier

# In[48]:


print("\033[5m"+ "Ada Boost Classifier"+"\033[0m\n")
AB = AdaBoostClassifier()
AB.fit(X_train,Y_train)


# In[49]:


#Model_com_df=pd.DataFrame(Model_comparison).T
#Model_com_df.pop(3)
#Model_com_df.columns=['Model Accuracy','Model F1-Score','CV Accuracy']
#Model_com_df=Model_com_df.sort_values(by='Model F1-Score',ascending=False)
#Model_com_df.style.format("{:.2%}").background_gradient(cmap='Blues')
##Model_com_df.style.format("{:.2%}")


# In[59]:


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef,     roc_auc_score,precision_score, recall_score,     average_precision_score

def get_scoring_functions():
    cfuncs = [accuracy_score, f1_score, matthews_corrcoef, recall_score]
    pfuncs = [roc_auc_score, average_precision_score]
    return cfuncs, pfuncs

def get_tuple_cols():
    cfuncs, pfuncs = get_scoring_functions()
    return ['model'] + [f.__name__ for f in cfuncs] + [f.__name__ for f in pfuncs]

def get_scores(model_name, y_true, y_preds):
    cfuncs, pfuncs = get_scoring_functions()

    cscores = {f.__name__: f(y_true, y_preds) for f in cfuncs}
    pscores = {f.__name__: f(y_true, y_preds) for f in pfuncs}

    d = {**cscores, **pscores}
    d['model'] = model_name

    return tuple([d[c] for c in get_tuple_cols()])

models = [Log,RF, DT, NB, SVM, Xg, SGD, AB]
model_names = [type(m).__name__ for m in models]

y_preds = {type(model).__name__: model.predict(X_test) for model in models}
#y_probs = {type(model).__name__: model.predict_proba(X_test)[:,1] for model in models}
scores = [get_scores(name, Y_test, y_preds[name]) for name in model_names]
df = pd.DataFrame(scores, columns=get_tuple_cols())
df=df.sort_values('f1_score',ascending=False).reset_index(drop=True)
df=df.style.background_gradient(cmap='viridis')
df


# In[ ]:




