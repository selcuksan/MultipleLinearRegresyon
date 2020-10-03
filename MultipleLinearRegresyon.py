import pandas as pd
import numpy as np

df = pd.read_csv("odev_tenis.csv")
print(df)

outlook = df.iloc[:,0:1].values
windy = df.iloc[:,3:4].values
play = df.iloc[:,4:5].values

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()
windy = ohe.fit_transform(windy).toarray()
play = ohe.fit_transform(play).toarray()
print(outlook)
print(windy)
print(play)

outlook_Df = pd.DataFrame(outlook,columns=["sunny","overcast","rainy"])
temp_Hum_Df = df.iloc[:,1:3]
windy_Df = pd.DataFrame(windy[:,0:1],columns=["windy"])
play_Df = pd.DataFrame(play[:,0:1],columns=["play"])

X = pd.concat([outlook_Df,temp_Hum_Df,windy_Df,play_Df],axis=1)
x= X.drop(["temperature"],axis=1)
y = df[["temperature"]]
print(pd.concat([x,y],axis=1))


from sklearn.linear_model import LinearRegression

lR = LinearRegression()
lR.fit(x,y)
tahmin = lR.predict(x)
print(tahmin)
print(y)


dftahmin = pd.DataFrame(lR.predict(x),columns=["tahmin degerleri"])
print(dftahmin)
dfGercek_degerler = y
sonuc = pd.concat([dftahmin,dfGercek_degerler],axis=1)
sonuc.columns=["tahmin degerleri","gercek degerler"]
print(sonuc)

import statsmodels.api as sm

X_l  = x.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(y,X_l).fit()
print(model.summary())

X_l  = x.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(y,X_l).fit()
print(model.summary())

X_l  = x.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(y,X_l).fit()
print(model.summary())


X1  = x.iloc[:,[0,1,2,3]]

lR2 = LinearRegression()
lR2.fit(X1,y)
tahmin = lR2.predict(X1)




sonDf = pd.DataFrame(tahmin,columns=["backward eliminationdan sonra tahminler"])
sonuc = pd.concat([dftahmin,dfGercek_degerler,sonDf],axis=1)
sonuc.columns=["tahmin degerler","gercek degerler","backward elimination sonrası tahminler"]
print(sonuc)