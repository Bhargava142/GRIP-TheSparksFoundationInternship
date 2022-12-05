#!/usr/bin/env python
# coding: utf-8

# # #Author : Bhargav G
# 

# ## GRIP @ The Sparks Foundation

# # Data Science And Business Analysis Internship
# 

# ### Task 1:Prediction Using Supervised ML with Python Scikit Learn

# ### Step 1:- Importing required libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn import metrics


# ### Step 2:- Gathering Data For Analysis

# In[2]:


url ="http://bit.ly/w-data"
data = pd.read_csv(url)
print("Read the data successfully.")
data.head()


# ### Step 3:- Discovering and Visualizing the data along with some statistical properties
# 
# 

# In[3]:


data.info()


# In[4]:


# Finding statistical properties of the data
data.describe()


# In[5]:


data.shape


# In[6]:


# Checking the existence of missing or null values
data.isnull().sum()


# ### Step 4:- Printing the Scatter plot to analyze the relationship between the variables

# In[7]:


plt.title('Hours vs Percentage')
sns.scatterplot(data=data, x='Hours', y='Scores')


# In[8]:


#Assumption check - checking the correlation between the data
data.corr()


# The data is 97% positively correlated

# ### Step 5:- Preprocessing the data

# In[9]:


X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[11]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[12]:


print('Intercept:',regressor.intercept_)
print('Coefficient:',regressor.coef_)


# ### Step 6:- Plotting the Line of Regression
# 

# In[13]:


#Plotting the regression line with the test data
line=regressor.coef_*X+regressor.intercept_
plt.title('Hours vs Percentage')
plt.scatter(X,y, color = 'purple')
plt.plot(X, line, color = 'orange')
plt.show()


# ###  Step 7:- Making Predictions

# In[15]:


y_pred = regressor.predict(X_test)


# In[16]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[17]:


#Making our predictions
hours = 9.25
h = np.array([hours])
h = h.reshape(-1,1)
mypred = regressor.predict(h)
print('for {} Hours of study, the student can score: {}.'.format(hours,mypred[0]))


# ### Step 8:- Model Evaluation

# In[20]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R 2 Score:', metrics.r2_score(y_test, y_pred))


# ## Thankyou

# In[ ]:




