#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.simplefilter("ignore")
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, roc_curve, auc


# In[2]:


train_id = pd.read_csv("train_identity.csv")
train_id.head()


# In[3]:


train_txn = pd.read_csv("train_transaction.csv")
train_txn.head()


# In[4]:


train_id.info()


# In[5]:


train_txn.info()


# In[6]:


test_id = pd.read_csv("test_identity.csv")
test_txn = pd.read_csv("train_transaction.csv")


# In[7]:


test_id.info()


# In[8]:


test_txn.info()


# In[9]:


train = train_txn.merge(train_id, how='left', on="TransactionID")
test = test_txn.merge(test_id, how='left', on="TransactionID")


# In[10]:


train.head()


# In[14]:


# find out missing values
train.isna().sum()/len(train) * 100


# In[15]:


# drop columns if missing values greater than 60%
for i in train.columns:
    if (train[i].isnull().sum() / len(train) * 100) > 60:
        train.drop(i, inplace = True, axis = 1)


# In[16]:


# filling 0 into na
for i in train.columns:
    train[i] = train[i].fillna(train[i].mode()[0])


# In[17]:


train.head()


# In[18]:


train.describe()


# In[19]:


train['isFraud'].value_counts()


# In[20]:


# visualize the amount of fraud or not
plt.subplots(figsize=(10,5))
sns.countplot(train['isFraud'], palette=["blue", "red"])
plt.show()
print(np.round(train[train['isFraud']==1].shape[0]/train.shape[0]*100,2),'% of train_data tested as fraud')
print(np.round(train[train['isFraud']==0].shape[0]/train.shape[0]*100,2),'% of test_data tested as not fraud')


# In[21]:


# visualize the fraud under ProductCD
plt.figure(figsize=(12,6))

train_ProductCD = (train.groupby(['isFraud'])['ProductCD']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('ProductCD'))
sns.barplot(x="ProductCD", y="percentage", hue="isFraud", data=train_ProductCD, palette=["blue", "red"]);


# In[22]:


# visualize fraud under card4 
plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
train_card4 = (train[~train['card4'].isnull()].groupby(['isFraud'])['card4']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('card4'))
sns.barplot(x="card4", y="percentage", hue="isFraud", data=train_card4, palette=["blue", "red"])
plt.title('Train')


# In[23]:


# figure out the correlation 
cor = train.corr()
rel = cor['isFraud'].sort_values(ascending = False)
rel


# In[24]:


sns.heatmap(cor)


# In[25]:


# select columns with relatively higher correlation
col = []
for i in range (len(rel)):
    if rel[i] > 0.1 or rel[i] < -0.1:
        col.append(rel.index[i])


# In[26]:


# find out the object data_type
tsf = []
for i in train.columns:
    if train.dtypes[i] == 'O':
        tsf.append(i)


# In[27]:


# convert category columns to int
la = LabelEncoder()
for i in tsf:
    train[i] = la.fit_transform(train[i])


# In[28]:


x = train[col]
x.drop('isFraud', inplace = True, axis = 1)
x.head()


# In[29]:


y = train['isFraud']
y


# In[85]:


del train


# In[30]:


# balancing
over = SMOTE()
x, y = over.fit_resample(x, y)
y.value_counts()


# In[31]:


x.describe()


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)


# In[33]:


rf = RandomForestClassifier(n_estimators = 200)
rf.fit(x_train, y_train)


# In[34]:


predicted = rf.predict(x_train)
print("Accuracy using Random Forest is {} %".format(accuracy_score(predicted, y_train)*100))


# In[35]:


predicted = rf.predict(x_test)
print("Accuracy using Random Forest is {} %".format(accuracy_score(predicted, y_test)*100))


# In[36]:


fpr, tpr, threshold= roc_curve(y_test, predicted, pos_label=1)
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.show()
print("AUC value is {} ".format(auc(fpr, tpr)))


# In[37]:


x.shape


# In[38]:


test.shape


# In[39]:


# checking missing values
for i in test.columns:
    if test[i].isnull().sum() / len(test) * 100 > 60:
        test.drop(i, inplace = True, axis = 1)


# In[40]:


# fill 0 into na
for i in test.columns:
    test[i] = test[i].fillna(test[i].mode()[0])
test.info()


# In[41]:


# figure out object columns
o = []
for i in test.columns:
    if test.dtypes[i] == 'O':
        o.append(i)
o


# In[42]:


# convert category to int
for i in o:
    test[i] = la.fit_transform(test[i])
test.info()


# In[43]:


# remove y from correlated variables
col.remove('isFraud')
col


# In[44]:


x_test_pre = test[col]
predicted = rf.predict(x_test_pre)
x_test_pre.shape


# In[45]:


# prediction result
result = pd.DataFrame( {'TransactionID':test['TransactionID'], 'isFraud':predicted})
result


# In[ ]:




