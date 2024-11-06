#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries and loading dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"D:\data science class\Machine learning\Fraud detection/Fraud.csv")


# In[2]:


dataset.head(20)


# In[3]:


dataset.info()


# In[4]:


dataset.isnull().sum()


# In[5]:


#Encoding the column type
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
dataset['type']=LE.fit_transform(dataset['type'])


# In[6]:


dataset['nameOrig']=LE.fit_transform(dataset['nameOrig'])
dataset['nameDest']=LE.fit_transform(dataset['nameDest'])


# In[7]:


dataset['isFraud'].value_counts()


# In[8]:


dataset.corr()


# In[9]:


dataset=dataset.drop(columns=['nameOrig','newbalanceOrig','nameDest','oldbalanceDest'],axis=1)


# In[10]:


dataset.corr()


# In[11]:


#visualizing the dataset to find the outliers
import seaborn as sns
import matplotlib.pyplot as plt
sns.boxplot(dataset)
plt.show()


# # Finding the skewness to decide the method for outlier treatment

# In[12]:


dataset['oldbalanceOrg'].skew()


# In[13]:


dataset['step'].skew()


# In[14]:


dataset['amount'].skew()


# In[15]:


dataset['newbalanceDest'].skew()


# In[16]:


dataset['isFlaggedFraud'].skew()


# In[17]:


dataset['isFraud'].skew()


# In[18]:


dataset.describe()


# In[19]:


#removing outliers using Inter Quartile Range in columns having high skewness
def remove_outliers_iqr(dataset, columns):
    for column in columns:
        Q1 = dataset[column].quantile(0.25)
        Q3 = dataset[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        dataset = dataset[(dataset[column] >= lower_bound) & (dataset[column] <= upper_bound)]
    return dataset

outlier_columns = ['oldbalanceOrg', 'amount', 'newbalanceDest', 'isFlaggedFraud']

dataset1 = remove_outliers_iqr(dataset,outlier_columns)


# In[20]:


dataset1.head()


# In[21]:


sns.boxplot(dataset1['newbalanceDest'])
plt.show()


# In[22]:


#removing outliers using Inter Quartile Range in columns having low skewness
def remove_outliers_z_score(dataset1, columns, threshold=3):
    for column in columns:
        mean = dataset1[column].mean()
        std = dataset1[column].std()
        z_scores = (dataset1[column] - mean) / std
        dataset1 = dataset1[np.abs(z_scores) <= threshold]
    return dataset1

column_to_remove_outlier = ['step']
clean_data = remove_outliers_z_score(dataset1, column_to_remove_outlier)


# In[23]:


clean_data.head()


# In[24]:


clean_data.corr()


# In[39]:


clean_data.head()


# # droping columns after treating columns for outliers based on corelation

# In[25]:


x=clean_data.drop(columns=['isFraud','newbalanceDest'])


# In[26]:


x.head()


# In[28]:


#naming 0 and 1
clean_data["isFraud"] = clean_data["isFraud"].map({0: "No Fraud", 1: "Fraud"})


# In[29]:


y=clean_data['isFraud']


# In[30]:


y


# In[32]:


# splitting the data in train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[33]:


#model building
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
clf = DecisionTreeClassifier()


# In[34]:


#pridicting the y_test values
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_pred


# In[35]:


#Evaluting the model
accuracy = accuracy_score(y_test, y_pred)
print( accuracy)


# In[36]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)


# In[37]:


#Finding the training data accuracy
bias=clf.score(x_train, y_train)
bias


# In[38]:


#finding the testing data accuracy
variance=clf.score(x_test,y_test)
variance


# In[40]:


features = np.array([[1,4,181.00,181.0,0]])
print(clf.predict(features))


# In[ ]:




