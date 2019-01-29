#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import packages
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix


# In[5]:


# Download the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None) 
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"]

# Check the data types
Mamm.dtypes


# In[6]:


# Corece to numeric and impute medians for BI-RADS column
Mamm.loc[:, "BI-RADS"] = pd.to_numeric(Mamm.loc[:, "BI-RADS"], errors='coerce')
# errors= xxx define whether to report there is an error when converting the data


# In[7]:


HasNan = np.isnan(Mamm.loc[:,"BI-RADS"])
Mamm.loc[HasNan, "BI-RADS"] = np.nanmedian(Mamm.loc[:,"BI-RADS"])


# In[8]:


# Check the distribution of the "BI-RADS" column
plt.hist(Mamm.loc[:, "BI-RADS"])
plt.hist(Mamm.loc[:, "BI-RADS"]);  # This way array will not be displayed


# In[9]:


# Replace outlier
TooHigh = Mamm.loc[:, "BI-RADS"] > 6


# In[10]:


Mamm.loc[TooHigh, "BI-RADS"] = 6
# Set outliers to 6


# In[11]:


# Check the distribution of the "BI-RADS" column
plt.hist(Mamm.loc[:, "BI-RADS"])


# In[12]:


# Corece to numeric and impute medians for Age column
Mamm.loc[:, "Age"] = pd.to_numeric(Mamm.loc[:, "Age"], errors='coerce')
HasNan = np.isnan(Mamm.loc[:,"Age"]) 
Mamm.loc[HasNan, "Age"] = np.nanmedian(Mamm.loc[:,"Age"])


# In[13]:


# Check the distribution of the "Age" column
plt.hist(Mamm.loc[:, "Age"])


# In[14]:


# The next ordinal or numeric column is >Density<. 
# Corece to numeric and impute medians for Density column
Mamm.loc[:, "Density"] = pd.to_numeric(Mamm.loc[:, "Density"], errors='coerce')
HasNan = np.isnan(Mamm.loc[:,"Density"])
Mamm.loc[HasNan, "Density"] =  np.nanmedian(Mamm.loc[:,"Density"])


# In[15]:


# Check the distribution of the "Density" column
plt.hist(Mamm.loc[:, "Density"]) 


# In[162]:


# Check the distribution of the "Severity" column
plt.hist(Mamm.loc[:, "Severity"]) 


# In[16]:


# Check the data types
Mamm.dtypes


# In[17]:


# Plot all the numeric columns against each other
scatter_matrix(Mamm) 


# In[18]:


_ = scatter_matrix(Mamm, c=Mamm.loc[:,"Severity"], figsize=[8,8], s=1000)
# s=1000 is the size of the dots 


# In[179]:


# Remove rows that contain one or more NaN
Mamm_FewerRows = Mamm.dropna(axis=0)
# Data.dropna(axis=0)
# dropna()
# axis=0 means drop that row of the data 


# In[180]:


# Check the first rows of the data frame
Mamm_FewerRows.head()


# In[181]:


# Check the number of rows and columns
Mamm_FewerRows.shape
##############


# In[182]:


# Remove columns that contain one or more NaN
Mamm_FewerCols = Mamm.dropna(axis=1)


# In[183]:


# Check the first rows of the data frame
Mamm_FewerCols.head()


# In[ ]:


# Check the number of rows and columns
Mamm_FewerCols.shape

