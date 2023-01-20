#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.optimize as opt
import errors as err

""" this function is used to read all the datas
read the datas from csv files 
co2 emission data and tottal population"""

def reading():
    data =pd.read_csv("/Users/karthikshivaprasad/Downloads/vyshakpgm/adsposter/co2.csv",skiprows=4,index_col=False)
    population =pd.read_csv("/Users/karthikshivaprasad/Downloads/vyshakpgm/adsposter/population.csv",skiprows=4,index_col=False)
    return data,population

"""ths function is used to taking data from a perticular year 
function used to taking data from yesr  1990 to 2015 """
def takingyear(plotonecountry):
    #taking only the rows  
    plotonecountryyr=plotonecountry.iloc[:,34:60]
    plotonecountryyr.reset_index(drop=True, inplace=True)
    year=[1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    #transpose the given data
    transpose=plotonecountryyr.transpose()
    plotresetonecountry=pd.DataFrame({'year':year,'value':transpose[0]})   
    return plotresetonecountry
"""this function used to normalize the given data 
function normalize the year and value from 0 to 1"""
def normalize(fornormalize):
    scaler = MinMaxScaler()
    #normalise the year column 
    scaler.fit(fornormalize[['year']])
    fornormalize['year'] = scaler.transform(fornormalize[['year']])
    #normalise the value column 
    scaler.fit(fornormalize[['value']])
    fornormalize['value'] = scaler.transform(fornormalize[['value']])
    return fornormalize

"""the function is used to do k means cluster
fuction div ve the data into  3 clusters """ 
def kmeansf(forkmeans):
    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(forkmeans[['year','value']])
    forkmeans['cluster']=y_predicted
    forkmeans.head()
    return  forkmeans,km
"""the function is used to plot the data 
in this function the given data is ploted in a scatterplot"""
def ploting(kmeansdata,km,countryname): 
    #taking values of diffrent cluster
    df1c = kmeansdata[kmeansdata.cluster==0]
    df2c = kmeansdata[kmeansdata.cluster==1]
    df3c = kmeansdata[kmeansdata.cluster==2]    
    plt.figure()
    plt.title("CO2 EMISSION OF "+countryname)
    plt
    plt.scatter(df1c['year'],df1c['value'],color='green',label='cluster1')
    plt.scatter(df2c['year'],df2c['value'],color='red',label='cluster2')
    plt.scatter(df3c['year'],df3c['value'],color='black',label='cluster3')
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
    plt.xlabel('YEAR')
    plt.ylabel('CO2 EMISSION')
    plt.legend()
    plt.show()
    
data,population=reading()
#taking  the co2 data with country name with China  
plotchina=data.loc[(data['Country Name']=='China') ]
#taking  the co2 data with country name with United Kingdom  
plotuk=data.loc[(data['Country Name']=='United Kingdom') ]
plotchina=data.loc[(data['Country Name']=='China') ]
#taking  the poulation  data with country name with China
populationplotch=population.loc[(population['Country Name']=='China') ]
#taking  the poulation  data with country name with United Kingdom
populationplotuk=population.loc[(population['Country Name']=='United Kingdom') ]
plotresetchina = takingyear(plotchina)
plotresetuk = takingyear(plotuk) 

populationrestch = takingyear(populationplotch)
populationresetuk = takingyear(populationplotuk)  
plt.figure()
plt.title("population of China and UK ")
#plot population of china with year
plt.plot(populationrestch['year'],populationrestch['value'],label='popultion of China')
#plot population of United Kingdom with year
plt.plot(populationresetuk['year'],populationresetuk['value'],label='popultion of United Kingdom')
plt.xlabel('year')
plt.ylabel('population')
plt.legend()
plt.show()
plotresetchina=normalize(plotresetchina)
kmeanschina,km=kmeansf(plotresetchina)
plotresetuk=normalize(plotresetuk)
kmeannsuk,km1=kmeansf(plotresetuk)
    
ploting(kmeanschina,km,countryname='CHINA')
ploting(kmeannsuk,km1,countryname='UUNITED KINGDOM')

""" calculate exponatial function withnv scale factor n0,and growthh rate g """
def exponential(t, n0, g):

    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f


populationrestch["year"] = pd.to_numeric(populationrestch["year"])
#fitting curve calculated the values for ploting fiting curve 
param, covar = opt.curve_fit(exponential, populationrestch["year"], populationrestch["value"],
p0=(73233967692.102798, 0.03))
plt.Figure()
plt.title("Fitting curve for  population of China ")
#add fiting colum in this given data frame
populationrestch["fit"] = exponential(populationrestch["year"], *param)
#this line is used to plot normal population graph of china 
plt.plot(populationrestch['year'],populationrestch['value'],label='population of China')
#this line is for  to  fiting plot  of  the  population  of china
plt.plot(populationrestch['year'],populationrestch['fit'],label='fitting curve')
plt.legend()
plt.show()

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    #in this line plot an elbow plot to undestand which cluster is better 
    km.fit(populationrestch[['year','value']])
    sse.append(km.inertia_)
plt.title("Elbow plot of China population")
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)

#finding sigma and forcast value o thegiven data
sigma = np.sqrt(np.diag(covar))
year1 = np.arange(1990, 2031)
forecast = exponential(year1, *param)
#find the error range of the forecast data
low1, up1 = err.err_ranges(year1, exponential, param, sigma)
plt.figure()
plt.plot(populationrestch["year"], populationrestch["value"], label="Population")
#plot the forecast data
plt.plot(year1, forecast, label="Forecast")
plt.fill_between(year1, low1, up1, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("Population")
plt.legend()
plt.show()


    