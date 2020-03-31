
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


# In[255]:


df = pd.read_csv('/Users/harshitgarg/Downloads/rainfall.csv', index_col=0) 
df1 = df.loc['LAKSHADWEEP'] #Selecting a particular state for rainfall characteristics
df1 = df1[['YEAR', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
       'OCT', 'NOV', 'DEC']]
df1.index.name=None
df1.reset_index(inplace=True)

df1 = df1.interpolate(method ='linear', limit_direction ='backward', limit = 100) #interpolating NaN values

start = datetime.datetime.strptime("1901-01-01", "%Y-%m-%d")
date_list = [start + relativedelta(months=x) for x in range(0,114*12)]  #indexing by month+year
df2 = pd.DataFrame(date_list,columns = ['TIME'])

zeroes = [0.00]*(114*12)
df1 = df1[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG',
       'SEP', 'OCT', 'NOV', 'DEC']]
df2['rain_mm'] = zeroes

k = 0
for i in range(114):
    for j in list(df1.iloc[i]):
        df2.at[k,'rain_mm'] = j
        k = k + 1
df2.set_index(['TIME'], inplace=True)
a = math.sqrt(np.var(df2['rain_mm']))
b = np.mean(df2['rain_mm'])
df2['rain_mm'] = (df2['rain_mm'] - b)/a  #standardizing the data


# In[268]:


df2.rain_mm.plot(figsize=(32,16), title= 'Monthly Rainfall', fontsize=14)
plt.savefig('month_ridership.png', bbox_inches='tight')              #Plot data


# In[257]:


decomposition = seasonal_decompose(df2.rain_mm,freq=12,extrapolate_trend = 1)  #decompose using additive method
fig = plt.figure()  
fig = decomposition.plot()  
fig.set_size_inches(15, 8)


# In[258]:


trend = decomposition.trend
seasonal = decomposition.seasonal 
print(trend)


# In[259]:


def test_stationarity(timeseries,lag):
    
    #Determing rolling statistics
    #pd.rolling_mean(timeseries, window=12)
    rolmean = timeseries.rolling(12).mean()   #moving average
    rolstd = timeseries.rolling(12).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag = lag)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
test_stationarity(df2.rain_mm,'AIC')
test_stationarity(df2.rain_mm,'BIC')


# In[260]:


#Though we have achieved stationarity we still show application of first difference and seasonal difference
#first difference
df2['first_difference'] = df2.rain_mm - df2.rain_mm.shift(1)  
test_stationarity(df2.first_difference.dropna(inplace=False),'AIC')


# In[261]:


#seasonal difference
df2['seasonal_difference'] = df2.rain_mm - df2.rain_mm.shift(12)  
test_stationarity(df2.seasonal_difference.dropna(inplace=False),'AIC')
print(df2)


# In[262]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df2.rain_mm.iloc[:], lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df2.rain_mm.iloc[:], lags=50, ax=ax2)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df2.first_difference.iloc[1:], lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df2.first_difference.iloc[1:], lags=50, ax=ax2)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df2.seasonal_difference.iloc[13:], lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df2.seasonal_difference.iloc[13:], lags=50, ax=ax2)


# In[263]:


mod = sm.tsa.statespace.SARIMAX(df2.rain_mm, trend='ct', order=(2,0,2), seasonal_order=(1,1,1,12),enforce_invertibility=False,enforce_stationarity=False)
results = mod.fit()
print(results.summary()) #model fit


# In[264]:


df2['forecast'] = results.predict(start = 1200, end= 1368, dynamic= True)  #comparing prediction with actual
df2[['rain_mm', 'forecast']].plot(figsize=(12, 8))  
plt.savefig('ts_df_predict.png', bbox_inches='tight')


# In[265]:


npredict = 168  #data zoomed
fig, ax = plt.subplots(figsize=(12,6))
npre = 12
ax.set(title='Rainfall', xlabel='Date', ylabel='Standardized Rainfall')
ax.plot(df2.index[-npredict-npre+1:], df2.ix[-npredict-npre+1:, 'rain_mm'], 'o', label='Observed')
ax.plot(df2.index[-npredict-npre+1:], df2.ix[-npredict-npre+1:, 'forecast'], 'g', label='Dynamic forecast')
legend = ax.legend(loc='lower right')
legend.get_frame().set_facecolor('w')
plt.savefig('ts_predict_compare.png', bbox_inches='tight')


# In[266]:


start = datetime.datetime.strptime("2015-01-01", "%Y-%m-%d")  #adding new data for predicition
date_list = [start + relativedelta(months=x) for x in range(0,12)]
future = pd.DataFrame(index=date_list, columns= df.columns)
df2 = pd.concat([df2, future])


# In[267]:


df2['forecast'] = results.predict(start = 1368, end = 1379, dynamic= True)  #forecasting for new data
df2[['rain_mm', 'forecast']].ix[-24:].plot(figsize=(12, 8)) 
plt.savefig('ts_predict_future.png', bbox_inches='tight')
