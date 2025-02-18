# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:13:59 2025

@author: grade
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import seaborn as sns
from statsmodels.formula.api import ols #to run multiple regression model

import warnings 

def outlier_imputation(dataframe,column):
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3-q1
    lower_bound = q1 - iqr * 1.5
    upper_bound = q3 + iqr * 1.5
    dataframe[column] = np.where(
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound),
        dataframe[column].median(),
        dataframe[column]
    )
    return dataframe
    
    

warnings.filterwarnings('ignore')

#import churn data

df = pd.read_csv(r"C:\Users\grade\Downloads\multiple linear regression house sales\realtor-data.zip.csv")
print(df.duplicated()) #check for duplicates if they exist
print(df.duplicated().value_counts()) #counts true and false if 1 true appears, duplicates exist
print(df.isnull().sum()) #check how many nulls exist in each column if they exist and prints out how many nulls exist in each column
print(df.shape)
#i identify many nulls
#i notice that the houses seem to come in large chunks of same state, i believe interpolating to nearest value to be best way to treat these nulls

#removal of columns that arent reall house characteristics
df.drop('brokered_by', axis=1, inplace=True)
df.drop('status',axis=1,inplace = True)
df.drop('street',axis=1,inplace = True)
df.drop('zip_code',axis=1,inplace = True)
df.drop('prev_sold_date',axis = 1,inplace = True)

#making boxplots of columns before interpolation 
plt.figure(1)
plt.title('price')
priceBP = df.boxplot(column = ['price'])
plt.figure(2)
plt.title('bed')
bedBP = df.boxplot(column = ['bed'])
plt.figure(3)
plt.title('bath')
bathBP = df.boxplot(column = ['bath'])
plt.figure(4)
plt.title('acre lot')
acre_lotBP = df.boxplot(column = ['acre_lot'])
plt.figure(5)
plt.title('house size')
house_sizeBP = df.boxplot(column = ['house_size'])

#interpolating columns with missing values
df['price'].interpolate(method='linear', inplace=True)
df['bed'].interpolate(method='linear',inplace = True)
df['bath'].interpolate(method='linear',inplace = True)
df['acre_lot'].interpolate(method='linear',inplace=True)
df['house_size'].interpolate(method='linear',inplace=True)

#boxplots of columns after interpolation
plt.figure(6)
plt.title('price interpolated')
priceBP_inter = df.boxplot(column = ['price'])
plt.figure(7)
plt.title('bed interpolated')
bedBP_inter = df.boxplot(column = ['bed'])
plt.figure(8)
plt.title('bath interpolated')
bathBP_inter = df.boxplot(column = ['bath'])
plt.figure(9)
plt.title('acre_lot interpolated')
acre_lotBP_inter = df.boxplot(column = ['acre_lot'])
plt.figure(10)
plt.title('house_size interpolated')
house_sizeBP_inter = df.boxplot(column = ['house_size'])


#all rows with a null state and city are removed since they are a very small amount relative to dataset
df.dropna(subset = ['city'],inplace = True)
df.dropna(subset = ['state'],inplace = True)
print(df.isnull().sum())
print(df.shape)

#outlier treatment
#replace outliers with median value to bring data closer together\

df = outlier_imputation(df, 'price')
df = outlier_imputation(df, 'bed')
df = outlier_imputation(df, 'bath')
df = outlier_imputation(df, 'acre_lot')
df = outlier_imputation(df, 'house_size')

#imputing prior outliers with median brings data much closer together
plt.figure(11)
plt.title('price imputed')
priceBP_inter = df.boxplot(column = ['price'])
plt.figure(12)
plt.title('bed imputed')
bedBP_inter = df.boxplot(column = ['bed'])
plt.figure(13)
plt.title('bath imputed')
bathBP_inter = df.boxplot(column = ['bath'])
plt.figure(14)
plt.title('acre_lot imputed')
acre_lotBP_inter = df.boxplot(column = ['acre_lot'])
plt.figure(15)
plt.title('house_size imputed')
house_sizeBP_inter = df.boxplot(column = ['house_size'])

#we have categorical data that needs to be converted into numerical data, namely city names and state names
#this will be done by imputing the median price of houses in each city, and median price of each house in states




city_median_map = df.groupby('city')['price'].median().to_dict() #gets median price of each city and creates a dictionary in which each city is mapped to median price
df['city'] = df['city'].map(city_median_map)#each city is replaced with median price

state_median_map = df.groupby('state')['price'].median().to_dict()
df['state'] = df['state'].map(state_median_map)

real_estate_price_model = ols("price ~ bed + bath + acre_lot + city + state + house_size", data=df).fit()
print(real_estate_price_model.params)
print(real_estate_price_model.summary())

#p - values all 0 will not be reduced

mse = real_estate_price_model.mse_resid
print("RMSE: "+str(np.sqrt(mse))+'\n') 

#RMSE is mean difference between predictions and atcua; values

residuals = real_estate_price_model.resid
predictions = real_estate_price_model.predict()
plt.figure(45)
sns.residplot(x=predictions,y=residuals,data = df)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")



