#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import scipy.optimize as opt


def reading():
    data =pd.read_csv("/Users/karthikshivaprasad/Downloads/vyshakpgm/adsposter/co2.csv",skiprows=4,index_col=False)
    population =pd.read_csv("/Users/karthikshivaprasad/Downloads/vyshakpgm/adsposter/population.csv",skiprows=4,index_col=False)
    return data,population

def takingyear(plotonecountry):
    plotonecountryyr=plotonecountry.iloc[:,34:60]
    plotonecountryyr.reset_index(drop=True, inplace=True)
    year=[1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015]
    transpose=plotonecountryyr.transpose()
    plotresetonecountry=pd.DataFrame({'year':year,'value':transpose[0]})   
    return plotresetonecountry

def normalize(fornormalize):
    scaler = MinMaxScaler()
    scaler.fit(fornormalize[['year']])
    fornormalize['year'] = scaler.transform(fornormalize[['year']])
    scaler.fit(fornormalize[['value']])
    fornormalize['value'] = scaler.transform(fornormalize[['value']])
    return fornormalize
 
def kmeansf(forkmeans):
    km = KMeans(n_clusters=3)
    y_predicted = km.fit_predict(forkmeans[['year','value']])
    forkmeans['cluster']=y_predicted
    forkmeans.head()
    return  forkmeans,km

def ploting(kmeansdata,km,countryname):
    
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
plotchina=data.loc[(data['Country Name']=='China') ]
plotuk=data.loc[(data['Country Name']=='United Kingdom') ]
populationplotch=population.loc[(population['Country Name']=='China') ]
populationplotuk=population.loc[(population['Country Name']=='United Kingdom') ]
plotresetchina = takingyear(plotchina)
plotresetuk = takingyear(plotuk) 

populationrestch = takingyear(populationplotch)
populationresetuk = takingyear(populationplotuk)  
plt.figure()
plt.plot(populationrestch['year'],populationrestch['value'],label='popultion of china')
plt.plot(populationresetuk['year'],populationresetuk['value'],label='popultion of united kingdom')
plt.xlabel('YEAR')
plt.ylabel('population')
plt.legend()
plt.show()
plotresetchina=normalize(plotresetchina)
kmeanschina,km=kmeansf(plotresetchina)
plotresetuk=normalize(plotresetuk)
kmeannsuk,km1=kmeansf(plotresetuk)
    
ploting(kmeanschina,km,countryname='CHINA')
ploting(kmeannsuk,km1,countryname='UUNITED KINGDOM')


def exponential(t, n0, g):

    t = t - 1960.0
    f = n0 * np.exp(g*t)
    return f

print(type(populationrestch["year"].iloc[1]))
populationrestch["year"] = pd.to_numeric(populationrestch["year"])
print(type(populationrestch["year"].iloc[1]))
param, covar = opt.curve_fit(exponential, populationrestch["year"], populationrestch["value"],
p0=(73233967692.102798, 0.03))

populationrestch["fit"] = exponential(populationrestch["year"], *param)
populationrestch.plot("year", ["value", "fit"])
plt.show()

    