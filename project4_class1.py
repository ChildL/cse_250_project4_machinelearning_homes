# %%
import pandas as pd 
import numpy as np
import seaborn as sns
import altair as alt
# %%
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
# %%
dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")   

# alt.data_transformers.enable('json')
# %%
dwellings_denver.head()
# %%
dwellings_neighborhoods_ml.head()
# %% 
dwellings_ml.head()
# %%
dwellings_ml.syear.describe()
# shows house sold multiple times
# %%
dwellings_ml.parcel.value_counts()
# data represents sale. Need to filter so it represents house not house/sale

# %%
# - https://scikit-learn.org/stable/modules/tree.html#classification
X_pred = dwellings_ml.drop(['before1980', 'yrbuilt'], axis = 1)
y_pred = dwellings_ml.filter(["before1980"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76)  
# %%
X_pred
# %%
y_pred
# %%
# shows all the columns
dwellings_ml.columns
# %%
x = dwellings_ml.filter(['livearea','numbaths'])
y = dwellings_ml['before1980']
# %%
print(dwellings_ml.shape)
print(x.shape)
print(y.shape)

# %%
x = dwellings_ml.filter(['livearea','numbaths'])
y = dwellings_ml['before1980']
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = .34, 
    random_state = 76) 


# %%
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# %%

y_test.head(10).mean()

# %%
x = dwellings_ml.filter(['sprice'])
y = dwellings_ml['before1980']
x_train, x_test, y_train, y_test = train_test_split(
    x, 
    y, 
    test_size = .34, 
    random_state = 76) 

# %%
x_train.head(10).mean()

# %%
a_subset = dwellings_ml.filter(["livearea", 'basement', 'numbaths']).sample(500)
# %%
b_subset = dwellings_ml.filter(['yrbuilt']).sample(500)
# %%
df1 = dwellings_ml[['basement', 'livearea', 'yrbuilt', 'stories', 'nocars', 'sprice', 'syear', 'before1980']]

df1 = df1.groupby('before1980').mean().reset_index()
# %%
# calculating sale price before 1980 and after 1980
df2 = df1.copy()
df2.avg_sprice = df2.sprice.mean
# %%
df2.avg_sprice