#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math 
import time
from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
import datetime
import operator
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


#loading the all three datasets
confirmed_cases=pd.read_csv("C:\\Users\\TUSHAR SAIN\\Downloads\\time_series_covid-19_confirmed.csv")
death_reported=pd.read_csv("C:\\Users\\TUSHAR SAIN\\Downloads\\time_series_covid-19_deaths.csv")
recovered_cases=pd.read_csv("C:\\Users\\TUSHAR SAIN\\Downloads\\time_series_covid-19_recovered.csv")


# In[5]:


confirmed_cases.head()


# In[6]:


death_reported.head()


# In[7]:


recovered_cases.head()


# In[20]:


#extracting only the dates columns that have information of confirmed ,deaths and recovered cases
confirmed=confirmed_cases.iloc[:,4:]
deaths=death_reported.iloc[:,4:]
recoveries=recovered_cases.iloc[:,4:]



# In[21]:


#check the head of the outbreak cases
confirmed.head()


# In[22]:


#finding the total confirmed cases,dath cases and the recovered cases and append them to an 4 empty lists
#also,calculate the total mortality rate which is the death_sum/confirmed cases
dates=confirmed.keys()
world_cases=[]
total_deaths=[]
mortality_rate=[]
total_recovered=[]
for i in dates:
    confirmed_sum=confirmed[i].sum()
    death_sum=deaths[i].sum()
    recovered_sum=recoveries[i].sum()
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)


# In[23]:


#lets display each of the newly created variables
confirmed_sum


# In[24]:


death_sum


# In[25]:


recovered_sum


# In[26]:


world_cases


# In[27]:


#converting all the dates ab=nd the cases into numpy arrays
days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases=np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)


# In[ ]:





# In[31]:


print(days_since_1_22)
print(world_cases)
print(total_deaths)
print(total_recovered)


# In[38]:


#future prediction
days_in_future=10
future_forecast=np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates=future_forecast[:-10]
print(future_forecast)


# In[43]:


#converting all the integers into datetime for better visualization
start='1/22/2020'
start_date=datetime.datetime.strptime(start,"%m/%d/%Y")
future_forecast_dates=[]
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date+datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
    


# In[41]:


#for the visualization with the data 
latest_confirmed=confirmed_cases[dates[-1]]
latest_deaths=death_reported[dates[-1]]
latest_recoveries=recovered_cases[dates[-1]]


# In[42]:


latest_confirmed


# In[44]:


latest_deaths


# In[45]:


latest_recoveries


# In[47]:


#unique countries
unique_countries=list(confirmed_cases['Country/Region'].unique())
unique_countries


# In[48]:


#total number of confirmed cases of each country
country_confirmed_cases=[]
no_cases=[]
for i in unique_countries:
    cases=latest_confirmed[confirmed_cases['Country/Region']==i].sum()
    if cases>0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
for i in no_cases:
    unique_countries.remove(i)
unique_countries=[k for k,v in sorted(zip(unique_countries,country_confirmed_cases),key=operator.itemgetter(1),reverse=True)]
for i in range(len(unique_countries)):
                  country_confirmed_cases[i]=latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()


# In[50]:


#number of cases per country
print("Confirmed cases by countries/Region")
for i in range(len(unique_countries)):
    print(f'{unique_countries[i]}:{country_confirmed_cases[i]}cases')


# In[52]:


#list of unique provinces
unique_provinces=list(confirmed_cases['Province/State'].unique())
#non provinance states
outliers=['United Kingdom','Denmark','France']
for i in outliers:
    unique_provinces.remove(i)
    


# In[53]:


#number of confirmed cases in the province,state or city
province_confirmed_cases=[]
no_cases=[]
for i in unique_provinces:
    cases=latest_confirmed[confirmed_cases['Province/State']==i].sum()
    if cases>0:
        province_confirmed_cases.append(cases)
    else:
        no_cases.append(i)
for i in no_cases:
    unique_provinces.remove(i)


# In[55]:


#number of cases in each province
for i in range(len(unique_provinces)):
    print(f'{unique_provinces[i]}:{province_confirmed_cases[i]}case')


# In[57]:


#handling the nan values
nan_indices=[]
for i in range(len(unique_provinces)):
    if type(unique_provinces[i])==float:
        nan_indices.append(i)
unique_provinces=list(unique_provinces)
province_confirmed_cases=list(province_confirmed_cases)
for i in nan_indices:
    unique_provinces.pop(i)
    province_confirmed_cases.pop(i)


# In[58]:


#bar graph to see the total confirmed cases across different countries
plt.figure(figsize=(32,32))
plt.barh(unique_countries,country_confirmed_cases)
plt.title('number of covid-19 confirmed cases in countries')
plt.xlabel('number of covid-19 confirmed cases')
plt.show()


# In[60]:


china_confirmed=latest_confirmed[confirmed_cases['Country/Region']=='China'].sum()
outside_mainland_china_confirmed=np.sum(country_confirmed_cases)-china_confirmed
plt.figure(figsize=(16,9))
plt.barh('mainland China',china_confirmed)
plt.barh('Outside Mainland China',outside_mainland_china_confirmed)
plt.title('Number of Confirmed Coronavirus Cases')
plt.show()


# In[61]:


#toatal cases in mainland china and outside of it
print("Outside Mainland china {} cases:".format(outside_mainland_china_confirmed))
print("Mainland china:{} cases".format(china_confirmed))
print("total:{} cases".format(china_confirmed+outside_mainland_china_confirmed))


# In[63]:


#showing 10 countries with the most confirmed cases
visual_unique_countries=[]
visual_confirmed_cases=[]
others=np.sum(country_confirmed_cases[10:])
for i in range(len(country_confirmed_cases[:10])):
    visual_unique_countries.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])
visual_unique_countries.append('Other')
visual_confirmed_cases.append(others)


# In[64]:


#visualize 10 countries
plt.figure(figsize=(32,18))
plt.barh(visual_unique_countries,visual_confirmed_cases)
plt.title("Number of covid-19 confirmed cases in countries/region",size=20)
plt.show()


# In[70]:


#pie chart to see the total confirmed cases in 10 different countries
c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(unique_countries))
plt.figure(figsize=(10,10))
plt.title("Covid-19 confirmed cases in countries outside of mainland china")
plt.pie(visual_confirmed_cases,colors=c)
plt.legend(visual_unique_countries,loc='best')
plt.show()


# In[71]:


#pie chart for countries outside china
c=random.choices(list(mcolors.CSS4_COLORS.values()),k=len(unique_countries))
plt.figure(figsize=(10,10))
plt.title("Covid-19 confirmed cases in countries outside of mainland china")
plt.pie(visual_confirmed_cases[1:],colors=c)
plt.legend(visual_unique_countries[1:],loc='best')
plt.show()


# In[79]:


#SVM model
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, world_cases, test_size=0.15, shuffle=False)
kernel=["poly","sigmoid","rbf"]
c=[0.01,0.1,1,10]
gamma=[0.01,0.1,1]
epsilon=[0.01,0.1,1]
shrinking=[True,False]
svm_grid={"kernel":kernel,"C":c,"gamma":gamma,"epsilon":epsilon,"shrinking":shrinking}

svm=SVR()
svm_search=RandomizedSearchCV(svm,svm_grid,scoring="neg_mean_squared_error",cv=3,return_train_score=True,n_jobs=-1,n_iter=40,verbose=1)
svm_search.fit(X_train_confirmed,y_train_confirmed)


# In[82]:


svm_search.best_params_


# In[83]:


svm_confirmed=svm_search.best_estimator_
svm_pred=svm_confirmed.predict(future_forecast)


# In[84]:


svm_confirmed


# In[85]:


svm_pred


# In[86]:


#against test data
svm_test_pred=svm_confirmed.predict(X_test_confirmed)
plt.plot(svm_test_pred)
plt.plot(y_test_confirmed)
print("MAE",mean_absolute_error(svm_test_pred,y_test_confirmed))
print("MSE",mean_squared_error(svm_test_pred,y_test_confirmed))


# In[89]:


#total number of corona virus cases
plt.figure(figsize=(20,20))
plt.plot(adjusted_dates,world_cases)
plt.title("number of coronavirus cases over time",size=30)
plt.xlabel("number since 1/22/2020",size=30)
plt.ylabel("number of cases",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[90]:


#cofirmed vs predicted cases
plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,world_cases)
plt.plot(future_forecast,svm_pred,linestyle="dashed",color="purple")
plt.title("number of coronasvirus cases over time",size=30)
plt.xlabel("days since 1/22/2020",size=30)
plt.ylabel("number of cases",size=30)
plt.legend(['confirmed cases',"svm predictions"])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[91]:


#prediction for the next 10 days using svm
print("svm future predicton:")
set(zip(future_forecast_dates[-10:],svm_pred[-10:]))


# In[98]:


#using the linear regression model
from sklearn.linear_model import LinearRegression
linear_model=LinearRegression(fit_intercept=True)
linear_model.fit(X_train_confirmed,y_train_confirmed)
test_linear_pred=linear_model.predict(X_test_confirmed)
linear_pred=linear_model.predict(future_forecast)
print("MAE:",mean_absolute_error(test_linear_pred,y_test_confirmed))
print("MSE:",mean_squared_error(test_linear_pred,y_test_confirmed))


# In[101]:


plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(["y_test_confirmed","Test_linear_pred"])


# In[106]:


plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,world_cases)
plt.plot(future_forecast,linear_pred,linestyle="dashed",color="orange")
plt.title("Number of coronavirus cases over time",size=30)
plt.xlabel("Days since 1/22/2020",size=30)
plt.ylabel("Number of cases",size=30)
plt.legend(["Confirmed cases","linear regression prediction"])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[107]:


#preediction for the next 10 dasys using linear rregression
print("Linear regression futureprediction:")
print(linear_pred[-10:])


# In[109]:


#toatal deaths over time
plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,total_deaths,color="red")
plt.title("Number of coronavirus deaths over time",size=30)
plt.xlabel("Time",size=30)
plt.ylabel("Number of deaths",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[111]:


mean_mortality_rate=np.mean(mortality_rate)
plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,mortality_rate,color="red")
plt.axhline(y=mean_mortality_rate,linestyle="--",color="black")
plt.title("Mortality Rate of coronavirus over Time",size=30)
plt.legend(["Mortality rate","y="+str(mean_mortality_rate)])
plt.xlabel("Time",size=30)
plt.ylabel("Mortality Rate",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[113]:


#coronavirus cases Recovered Over time
plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,total_recovered,color="green")
plt.title("Number of coronavirus cases Recovered over time",size=30)
plt.xlabel("Time",size=30)
plt.ylabel("Number of deaths",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[116]:


#coronavirus cases Recovered vs deaths 
plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,total_deaths,color="r")
plt.plot(adjusted_dates,total_recovered,color="b")
plt.legend(["Death","Recoveries"],loc="best",fontsize=30)
plt.title("Number of coronavirus cases",size=30)
plt.xlabel("Time",size=30)
plt.ylabel("Number of deaths",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[118]:


#coronavirus cases Recovered vs deaths 
plt.figure(figsize=(20,12))
plt.plot(total_recovered,total_deaths,color="g")
plt.title("Coronavirus Deaths vs Cronavirus Recoveries",size=30)
plt.xlabel("Total number of Coronavirus Recoveries",size=30)
plt.ylabel(" Total Number of Coronavirus deaths",size=30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[ ]:




