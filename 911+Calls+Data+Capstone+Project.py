
# coding: utf-8

# # 911 Calls Capstone Project

# I have analysed 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). 
# 
# The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# I have done some modifications in dataset to make it suitable for EDA.
# * Converted lat,lng,zip columns to numerical format.
# * I have not used all the entries from the original dataset.

# ## Data and Setup

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[2]:

df=pd.read_csv('911.csv')


# In[3]:

df.head()


# ** Get general info of dataset and check for null values **

# In[4]:

df.info()


# ## EDA

# ** Top 5 zipcodes for 911 calls **

# In[5]:

df['zip'].value_counts().head(5)


# ** Top 5 townships (twp) for 911 calls **

# In[6]:

df['twp'].value_counts().head(5)


# ** Number of unique values in 'title' of emergency **

# In[7]:

df['title'].nunique()


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[8]:

df['Reason']=df['title'].apply(lambda x: x.split(':')[0])
df['title']=df['title'].apply(lambda x: x.split(':')[1])


# In[9]:

df.head()


# In[10]:

df['Reason'].unique()


# In[11]:

df['title'].nunique()


# ** So, there are 3 unique Reasons and 81 unique titles for emergency.**

# ** Most common Reason for a 911 call **

# In[12]:

df['Reason'].value_counts().head(5)


# In[13]:

sns.countplot(x=df['Reason'])


# ** Top 5 titles of emergency calls. **

# In[14]:

df['title'].value_counts()[:5]


# In[15]:

plt.figure(figsize=(8,6))
df.groupby('title').count().sort_values('e',ascending=0)[:5]['e'].plot(kind='barh')


# ** Plot of top 5 townships calling 911**

# In[16]:

plt.figure(figsize=(8,6))
df.groupby('twp').count().sort_values('e',ascending=0)[:5]['e'].plot(kind='barh')


# **Plot of top 5 townships with reasons**

# In[17]:

l=list(df['twp'].value_counts()[:5].index)


# In[18]:

plt.figure(figsize=(8,6))
sns.countplot(data=df[df['twp'].apply(lambda x:(True if x in l else False))],x='twp',hue='Reason')


# In[ ]:




# ** 'timeStamp' column is in str format. So converted it to TimeStamp format.**

# In[19]:

df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# ** Created new columns Hour, Month,Day of week and Date from timeStamp column. **

# In[20]:

df['Hour']=df['timeStamp'].apply(lambda x: x.hour)


# In[21]:

df['Month']=df['timeStamp'].apply(lambda x:x.month)


# In[22]:

df['Day of Week']=df['timeStamp'].apply(lambda x: x.dayofweek)


# In[23]:

df['Date']=df['timeStamp'].apply(lambda x: x.date())


# In[24]:

df.head()


# **Coverting 'Day of Week' from numbers to words**

# In[25]:

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[26]:

df['Day of Week']=df['Day of Week'].map(dmap)


# In[27]:

df.head()


# In[ ]:




# ** Countplot of the Day of Week column with the hue of the Reason column. **

# In[28]:

plt.figure(figsize=(8,6))
sns.countplot(x='Day of Week',data=df,hue='Reason')
plt.legend(loc='upper left',bbox_to_anchor=(1,1))


# From above chart, it can be seen that number of emergencies are least on Sundays especially the 'Traffic' kind of emergencies.

# ** Countplot of the Day of Week column with the hue of the Reason column. **

# In[29]:

plt.figure(figsize=(8,6))
sns.countplot(x='Month',data=df,hue='Reason')
plt.legend(loc='upper left',bbox_to_anchor=(1,1))


# In[ ]:




# ** There is something strange in this plot. 9, 10 and 11th months are missing from the data.
# I tired using groupby method and lineplot to get rough estimate of the missing months **

# In[30]:

a=df.groupby(df['Month']).count()
a.head()


# In[31]:

plt.figure(figsize=(8,6))
a['e'].plot()


# From above plot as well as the bar graph, it can be seen that number of calls are decreasing towrands year-end and are least in December.

# ** Using seaborn's lmplot() to create a linear fit on the number of calls per month.**

# In[39]:

a.drop('Month',axis=1,inplace=True)
a.reset_index(inplace=True)
a.head()


# In[40]:

sns.lmplot(x='Month',y='e',data=a)


# ** Plotting 911 call counts 'Date' column**

# In[41]:

plt.figure(figsize=(8,6))
bydate=df.groupby('Date').count()
bydate['e'].plot()
plt.ylabel('Number of calls')


# ** Recreated above plot as 3 separate plots with each plot representing a Reason for the 911 call**

# In[42]:

plt.figure(figsize=(8,6))
df[df['Reason']=='Traffic'].groupby('Date').count()['e'].plot()
plt.title('Traffic')
plt.ylabel('Number of calls')
plt.tight_layout()


# In[43]:

plt.figure(figsize=(8,6))
df[df['Reason']=='Fire'].groupby('Date').count()['e'].plot()
plt.title('Fire')
plt.ylabel('Number of calls')
plt.tight_layout()


# In[44]:

plt.figure(figsize=(8,6))
df[df['Reason']=='EMS'].groupby('Date').count()['e'].plot()
plt.title('EMS')
plt.ylabel('Number of calls')
plt.tight_layout()


# ** Created a heatmap of 'Day of Week','Hour' and count of 911 calls.**

# In[45]:

dayHour = df.groupby(by=['Day of Week','Hour']).count()['e'].unstack()
dayHour.head()


# In[46]:

plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')


# From above heatmap, it becomes easy to get idea how 911 calls are distributed among 'Hours' and 'Day of Week'. Call count is high in the evening esp, at 17th hour. Also, weekends show less call counts than weekdays.

# ** Created a clustermap using above dataframe so that similat patches of heatmap are arranged together. **

# In[47]:

sns.clustermap(dayHour,cmap='viridis')

