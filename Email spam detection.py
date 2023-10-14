#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv(r'C:\Users\nitip\OneDrive\Desktop\Data Science Project\spam.csv',encoding = "ISO - 8859 - 1")


# In[5]:


df


# In[6]:


df = df[['v1','v2']]
df.columns = ['Category','Message']
df


# In[7]:


df.groupby('Category').describe()


# In[8]:


encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])
df


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(df.Message, df.Category, test_size = 0.2)


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer 
vec = CountVectorizer()
x_train_count = vec.fit_transform(x_train.values)
x_train_count.toarray()


# In[13]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count,y_train)


# In[14]:


x_test_count = vec.transform(x_test)
accuracy = model.score(x_test_count, y_test)
print(f'Accuract of the model: {accuracy*100:.2f}%')


# In[15]:


def predict():
    message = input('Enter the message to predict: ')
    message = [message]
    vector = vec.transform(message)
    if (model.predict(vector)==1):
        print("Spam Email")
    else:
        print("Not Spam")


# In[16]:


predict()


# In[ ]:




