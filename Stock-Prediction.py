#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math


# In[2]:


df=pd.read_csv("tesla.csv")
df.info()


# In[3]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date',inplace=True)


# In[4]:


df.shape


# In[5]:


plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'], color='red')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD', fontsize = 18)
plt.show()


# In[6]:


data = df.filter(['Close'])
dataset = data.values #convert the data frame to a numpy array
training_data_len = math.ceil(len(dataset)*.8)  # number of rows to train the model on
training_data_len


# In[7]:


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
scaled_data


# In[8]:


train_data = scaled_data[0:training_data_len, :]
#Split the data into x_train, y_train datasets
x_train = []
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])
    if i<=60:
        print(x_train)
        print(y_train)
        print()


# In[9]:


x_train,y_train = np.array(x_train), np.array(y_train)


# In[10]:


x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[11]:


model =Sequential()
model.add(LSTM(64,return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(64, return_sequences= False))
model.add(Dense(32))
model.add(Dense(1))


# In[12]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[13]:


model.fit(x_train,y_train, batch_size=1, epochs=10)


# In[14]:


test_data= scaled_data[training_data_len-60:, :]
#create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])


# In[15]:


x_test = np.array(x_test)


# In[16]:


x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
x_test.shape


# In[17]:


predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[18]:


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visialization the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price' ,fontsize=18)
plt.plot(train['Close'],linewidth=3.5)
plt.plot(valid[['Close','Predictions']],linewidth=3.5)
plt.legend(['Train','Valid','Predictions'], loc='upper center')


# In[19]:


rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse


# In[20]:


valid


# In[21]:


tesla_quote = pd.read_csv('tesla.csv')
#Create new data frame
new_df = tesla_quote.filter(['Close'])
#get the last 60 days closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#scaled the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list
X_test = []
#append the past 60 days 
X_test.append(last_60_days_scaled)
#convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))
#get the predicted scaled price
pred_price= model.predict(X_test)
#undo the scalling
pred_price = scaler.inverse_transform(pred_price)
pred_price


# In[22]:


plt.plot(valid['Close'], label='Actual') 

# Plot the predicted values
plt.plot(valid['Predictions'], label='Predicted')

plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

plt.show()

