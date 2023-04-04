#!/usr/bin/env python
# coding: utf-8

# ## Mobile Price Prediction 

# ### In this project we are going to do perdict the house price based on some of the feature.Ok let's get start the prediction!!!

# ### Import the Libraires and Dataset

# In[60]:


# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[61]:


# Import Dataset
df = pd.read_csv("D:\Flask\FlaskMarket\market\data.csv")
df.head()


# ### Data Preprocessing

# In[62]:


# Drop the Unwannted column the first one
df.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[63]:


# After removing the dataset look like
df.head()


# In[64]:


# Checking null values into the dataset
df.isnull().sum()


# In[65]:


# Seeing the null values using heatmap
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[66]:


# Summary of the dataset
df.info()


# In[67]:


df.dtypes


# In[68]:


# Describe
df.describe()


# In[69]:


# Dropping the Name columns
df.drop(['Brand me'],axis=1,inplace=True)


# ### Handling Missing Values

# In[70]:


# We should handling these null or missing values
df.isnull().sum()


# In[71]:


# Fill up the mean values of all the missing value columns into the dataset
df['Ratings'].fillna(df['Ratings'].mean(),inplace = True)
df['RAM'].fillna(df['RAM'].mean(),inplace = True)
df['ROM'].fillna(df['ROM'].mean(),inplace = True)
df['Mobile_Size'].fillna(df['Mobile_Size'].mean(),inplace = True)
df['Selfi_Cam'].fillna(df['Selfi_Cam'].mean(),inplace = True)


# In[72]:


# After handling the all of the missing and null values from the dataset
df.isnull().sum()


# In[73]:


# We can able to see the there is no null values  
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[74]:


# Changing the Datatype
df['RAM'] = df['RAM'].astype('int64')
df['ROM'] = df['ROM'].astype('int64')
df['Selfi_Cam'] = df['Selfi_Cam'].astype('int64')


# In[75]:


# Final dataset for build a model
df.head()


# ### Exploratory Data Analysis 

# In[76]:


# Columns


# #### Let's Understand about the Features

# - **1. Brand me** This is first feature of our dataset. It's Denotes name of the mobile phones and   Brands.</br>
# - **2.Ratings** This Feature Denotes Number of the ratings gave by the consumers for each mobile.</br>
# - **3.RAM** It's have RAM size of the phone. </br>
# - **4.ROM** It's have ROM **(Internal Memory)** size of the phone. </br>
# - **5.Mobile_Size** It's represents how many inches of the particular mobile phone have. Here all the values are gave in **inches** </br>
# - **6.Primary_Cam** It's Denotes Number of the pixels of the primary camera **(Back Camera)** for each mobiles.</br>
# - **7.Selfi_Cam** It's Denotes Number of the pixels of the Selfi camera **(Front Camera)** for each mobiles.</br>
# - **8.Battery_Power** It's Denotes amount of the battery power in each mobiles in **mAh**.</br>
# - **9.Price** It's a Dependent Feature of the dataset. It's just denote prices of the each mobiles.
# 

# In[77]:


# Finding out the correlation between the features
corr = df.corr()
corr.shape


# In[78]:


# Plotting the heatmap of correlation between features
# plt.figure(figsize=(12,12))
# sns.heatmap(corr, cbar=False, square= True, fmt='.2%', annot=True, cmap='Greens')


# In[79]:


# plt.figure(figsize=(15,10))
# sns.set_style('whitegrid')
# sns.countplot(x='Ratings',data=df)


# In[80]:


# plt.figure(figsize=(15,10))
# sns.set_style('whitegrid')
# sns.countplot(x='RAM',data=df)


# In[81]:


# plt.figure(figsize=(15,10))
# sns.set_style('whitegrid')
# sns.countplot(x='ROM',data=df)


# # In[82]:


# plt.figure(figsize=(15,10))
# sns.set_style('whitegrid')
# sns.countplot(x='Primary_Cam',data=df)


# In[83]:


# plt.figure(figsize=(15,10))
# sns.set_style('whitegrid')
# sns.countplot(x='Selfi_Cam',data=df)


# In[84]:


# sns.distplot(df['RAM'].dropna(),kde=False,color='darkred',bins=10)


# In[85]:


# sns.distplot(df['Battery_Power'].dropna(),kde=False,color='green',bins=10)


# In[86]:


# sns.distplot(df['Price'].dropna(),kde=False,color='darkblue',bins=15)


# In[87]:


# sns.distplot(df['Battery_Power'].dropna(),kde=False,color='darkblue',bins=15)


# In[88]:


# plt.figure(figsize=(10,10))
# sns.pairplot(data=df)


# ## Feature Selection

# In[89]:


# Lets try to understand which are important feature for this dataset
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[90]:


X = df.iloc[:,1:7]  # Independent columns
y = df.iloc[:,[-1]] # Yarget column i.e price range 


# In[91]:


# Apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)


# In[92]:


dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)


# In[93]:


# Concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns


# In[94]:


featureScores


# In[95]:


print(featureScores.nlargest(4,'Score'))  #print 5 best features


# ### Feature Importance

# In[96]:


# Fiting Feature Seclection using Ensemble Methods
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)


# In[97]:


print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers


# In[98]:


# Plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
# plt.show()


# In[99]:


df.head()


# ## Model Fitting 

# ### Random Forest Regressor 

# In[100]:


# Value Assigning
X = df.iloc[:,[6,2,4,5,1,3]]
y = df.iloc[:,[-1]]


# In[101]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=15)


# In[102]:


from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor()
reg.fit(X_train,y_train)


# In[103]:


y_pred = reg.predict(X_train)


# In[104]:


# Training Score
# print("Training Accuracy:",reg.score(X_train,y_train)*100)


# In[105]:


# Testing Score
# print("Testing Accuracy:",reg.score(X_test,y_test)*100)


# In[106]:


# Visualizing the differences between actual prices and predicted values
# plt.scatter(y_train, y_pred)
# plt.xlabel("Prices")
# plt.ylabel("Predicted prices")
# plt.title("Prices vs Predicted prices")
# plt.show()


# In[107]:


# Sample Prediction
reg.predict([[4.0,128.0,6.00,48,13.0,4000]])


# ### Finally We Made it!!!
# 
# #### Random Forest Regressor
# 
# - **Tarining Accuracy:** 96.2% Accuracy <br/>
# - **Testing Accuracy:** 95.3% Accuracy
# 
# 

# In[115]:


# print("Enter mobile brand")
# b = input()
# print("Enter mobile ratings")
# rat = float(input())
# print("Enter mobile RAM")
# Ram = int(input())
# print("Enter mobile ROM")
# Rom = int(input())
# print("Enter mobile size")
# s = float(input())
# print("Enter primary camera pixels")
# pc = int(input())
# print("Enter selfie camera pixels")
# sc = int(input())
# print("Enter mobile battery power")
# bp = int(input())


# In[117]:

print(reg.predict([[4,128.0,6,48,13.0,4000]]))
# print(reg.predict([[Ram, Rom, s, pc, sc, bp]]))


# In[ ]:




