# %%
import pandas as pd 
import numpy as np
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from vega_datasets import data
# %%
from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# %%
dwellings_denver = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_denver/dwellings_denver.csv")
dwellings_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_ml/dwellings_ml.csv")
dwellings_neighborhoods_ml = pd.read_csv("https://github.com/byuidatascience/data4dwellings/raw/master/data-raw/dwellings_neighborhoods_ml/dwellings_neighborhoods_ml.csv")   

alt.data_transformers.enable('json')
# %%
# preliminary investigating the data
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
dwellings_ml.columns

# %%
# code from Hathaway class
h_subset = dwellings_ml.filter(['livearea', 'finbsmnt', 
    'basement', 'yearbuilt', 'nocars', 'numbdrm', 'numbaths', 
    'stories', 'yrbuilt', 'before1980']).sample(500)


sns.pairplot(h_subset, hue = 'before1980')

corr = h_subset.drop(columns = 'before1980').corr()
# %%
sns.heatmap(corr)
# %%
# Grand Question 1
alt.data_transformers.enable('data_server')
# %%
dwellings_denver.arcstyle.unique()
# %%
# Grand 1 Chart 1
arc_chart = (alt.Chart(dwellings_denver)
.mark_boxplot(
size = 50
)
.encode(
x = 'arcstyle',
y = alt.Y('yrbuilt', scale=alt.Scale(zero=False)),
color=alt.Color('arcstyle', legend=None))
.properties(
    width = 700
    )
)
arc_chart
# %%
arc_chart.save("arc_chart.svg")
# %%
# info for Grand 1 Chart 2
dwellings_ml.numbaths.value_counts()
# %%
# Grand 1 Chart 2
bath_df = dwellings_ml.filter(['numbaths', 'yrbuilt'])
# %%
bath_chart = (alt.Chart(bath_df)
.mark_boxplot(
    size = 40
)
    .encode(
    x = alt.X('numbaths:O'),
    y = alt.Y('yrbuilt', scale = alt.Scale(zero=False)),
    color=alt.Color('numbaths'))
    .properties(
    width = 800
    )    
)
bath_chart
# %%
bath_chart.save("bath_chart.svg")

# %%
# Grand 1 Chart 3
dwellings_ml.livearea.value_counts()
# %%
# didn't end up using this
livearea_df = dwellings_ml.filter(['livearea', 'before1980'])
# %%
h_subset = dwellings_ml.filter(['livearea', 'quality_C', 'quality_D', 'quality_X', 'stories', 'yrbuilt',]).sample(500)

corr = h_subset.corr()

# %%
dwelling_heatmap = sns.heatmap(data = corr, cmap = "YlGnBu")

plt.tight_layout()

dwelling_heatmap1 = dwelling_heatmap.get_figure()

dwelling_heatmap1.savefig("dwelling_heatmap.jpg")



# %%
# Grand Question 2
# RAN CODE WITH EVERYTHING TO FIGURE OUT WHICH FEATURES ARE OF IMPORTANCE GREATER THAN .20
# 

x_pred = dwellings_ml.filter(['arcstyle_ONE-STORY','gartype_Att','quality_C', 'livearea', 'basement', 'stories', 'netprice', 'numbdrm', 'finbsmt', 'numbaths', 'nocars', 'arcstyle_SPLIT LEVEL'])
y_pred =  dwellings_ml['before1980']       

X_train, X_test, y_train, y_test = train_test_split(
    x_pred, 
    y_pred, 
    test_size = .34, 
    random_state = 76)

# %%
# create the model
clf = GradientBoostingClassifier()

# train the model, give feature and target
clf = clf.fit(X_train, y_train)

# make predictions
y_predictions = clf.predict(X_test)

# test how accurate predictions are
metrics.accuracy_score(y_test, y_predictions)

# %%
# tried this it was not a good fit model
from sklearn.neighbors import KNeighborsClassifier

# finally worked
from sklearn.ensemble import RandomForestClassifier
# %%
clf = RandomForestClassifier(random_state = 10)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)
metrics.accuracy_score(y_test,y_pred)

# %%
# This code helps figure out the feature importance 
df_features = pd.DataFrame(
    {'f_names': X_train.columns, 
    'f_values': clf.feature_importances_}).sort_values('f_values', ascending = False)

df_features    

# %%
G3_chart = alt.Chart(df_features.query("f_values > .02"), width = 300, title = "Important Feature Chart").encode(alt.X('f_names', sort = "-y"), alt.Y('f_values')).mark_bar()
G3_chart
# %%
G3_chart.save("G3_chart.svg")
# %%
G3_Feature_chart = df_features.query("f_values > .02").plot.bar(x = 'f_names', y = 'f_values', title = "Important Feature Chart")

G3_Feature_chart
# %%
# couldn't get sizing right for labels
plt.figure(figsize=(12,18))
plt.tight_layout()
G3_Feature_chart = G3_Feature_chart.get_figure()
G3_Feature_chart.savefig("G3_Feature_chart.jpg")

# %%
# Grand Question 4
table = print(metrics.confusion_matrix(y_test, y_pred))
metrics.plot_confusion_matrix(clf, X_test, y_test)
# %%
metric_table = print(metrics.classification_report(y_test, y_pred))
#table = metric_table.to_markdown()
# %%
report = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict=True)).T
df = pd.DataFrame(report).transpose()
print(df.to_markdown())
# %%
# accuracy of the model
Accuracy = (4562 + 2622)/(262 + 345 + 4562+ 2622)

Accuracy
# %%
# precision-recall
precision = 4562/ (4562 + 262)
precision
# %%
recall = 4562/(4562 + 345)
recall


# %%
metrics.plot_roc_curve(clf, X_test, y_test)
