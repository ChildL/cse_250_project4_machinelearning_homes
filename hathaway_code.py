# %%
# import sys
# !{sys.executable} -m pip install seaborn scikit-learn
# %%
# the full imports
import pandas as pd 
import numpy as np
import seaborn as sns
import altair as alt

# the from imports
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
dwellings_ml.syear.describe()
dwellings_ml.parcel.value_counts()
# data represents sales.  Need to filter so it represents house not house/sale.

# %%
h_subset = dwellings_ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980']).sample(500)

sns.pairplot(h_subset, hue = 'before1980')

corr = h_subset.drop(columns = 'before1980').corr()
# %%
sns.heatmap(corr)

# %%
# - https://scikit-learn.org/stable/modules/tree.html#classification
X_pred = dwellings_ml.drop(['before1980', 'yrbuilt', 'parcel'], axis = 1)
y_pred = dwellings_ml.filter(["before1980"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76)  
    
# %%
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)
# %%
print(metrics.confusion_matrix(y_test, y_pred))
metrics.plot_confusion_matrix(clf, X_test, y_test)
# %%
metrics.plot_roc_curve(clf, X_test, y_test)

# %%
df_features = pd.DataFrame(
    {'f_names': X_train.columns, 
    'f_values': clf.feature_importances_}).sort_values('f_values', ascending = False)

df_features    

# %%
alt.Chart(df_features.query("f_values > .02")).encode(alt.X('f_values'), alt.Y('f_names', sort = "-x")).mark_bar()
# %%
df_features.query("f_values > .02").plot.bar(x = 'f_names', y = 'f_values')
# %%
X_pred = dwellings_ml.filter(df_features.query("f_values > .02").f_names.to_list(), axis = 1)
y_pred = dwellings_ml.filter(["before1980"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76)  

# %%
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)
# %%
print(metrics.confusion_matrix(y_test, y_pred))
metrics.plot_confusion_matrix(clf, X_test, y_test)

# %%
print(metrics.classification_report(y_test, y_pred))
# %%
metrics.plot_roc_curve(clf, X_test, y_test)
# %%
