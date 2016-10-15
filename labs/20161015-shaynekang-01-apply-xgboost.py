
# coding: utf-8

# # Apply XGboost
# 
# In this notebook, we're going to train our model to use XGBoost. All hyperparameters are inspired by [Sergey Lebedev's solution](https://www.kaggle.com/sergeylebedev/sf-crime/initial-benchmark-need-tuning/run/163881).

# In[1]:

import numpy as np
import pandas as pd


# ## Load datasets

# In[2]:

data = pd.read_csv("../data/train.csv", parse_dates=["Dates"])

data.drop("Resolution", axis=1, inplace=True)
data.drop("Descript", axis=1, inplace=True)

print(data.shape)
data.head(3)


# ## Feature Engineering

# ### DayOfWeek

# In[3]:

print(data["DayOfWeek"].unique())

day_of_week_dataframe = pd.get_dummies(data["DayOfWeek"], prefix="DayOfWeek").astype(np.bool)

data = pd.concat([data, day_of_week_dataframe], axis=1)
data.drop("DayOfWeek", axis=1, inplace=True)

print(data.shape)
data.head(3)


# ### PdDistrict

# In[4]:

print(data["PdDistrict"].unique())

pd_district_dataframe = pd.get_dummies(data["PdDistrict"], prefix="PdDistrict").astype(np.bool)

data = pd.concat([data, pd_district_dataframe], axis=1)
data.drop("PdDistrict", axis=1, inplace=True)

print(data.shape)
data.head(3)


# ### Dates

# In[5]:

def get_season(x):
    summer=0
    fall=0
    winter=0
    spring=0
    if (x in [5, 6, 7]):
        summer=1
    if (x in [8, 9, 10]):
        fall=1
    if (x in [11, 0, 1]):
        winter=1
    if (x in [2, 3, 4]):
        spring=1
    return summer, fall, winter, spring


# In[6]:

data["Dates_year"] = data["Dates"].dt.year
data["Dates_month"] = data["Dates"].dt.month
data["Dates_day"] = data["Dates"].dt.day
data["Dates_hour"] = data["Dates"].dt.hour
data["Dates_minute"] = data["Dates"].dt.minute
data["Dates_second"] = data["Dates"].dt.second
data["Awake"] = data["Dates_hour"].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
data["Summer"], data["Fall"], data["Winter"], data["Spring"]=zip(*data["Dates_month"].apply(get_season))

data.drop("Dates", axis=1, inplace=True)

print(data.shape)
data.head(3)


# ### Define a new feature named 'Address_Type' (Block/CrossRoad)

# In[7]:

from sklearn.preprocessing import LabelEncoder

data["Address_Type"] = np.nan

data.loc[data["Address"].str.contains("Block of"), "Address_Type"] = "Block"
data.loc[data["Address"].str.contains("/"), "Address_Type"] = "CrossRoad"

encoder = LabelEncoder()
data["Address_Type(encode)"] = encoder.fit_transform(data["Address_Type"])

data.head(3)


# ### One hot encode address

# In[8]:

# 누적값이 100개 이하인 경우는 'Others'로 바꾼다.
address_counts = data["Address"].value_counts()
other_index = address_counts[address_counts < 100].index
data.loc[data["Address"].isin(other_index), "Address"] = "Others"

print("The number of address types = {address}".format(address=len(data["Address"].value_counts())))
print(data.shape)
data.head()


# In[9]:

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(dtype=np.bool)

data["Address(encode)"] = label_encoder.fit_transform(data["Address"])
address = one_hot_encoder.fit_transform(data[["Address(encode)"]])

print(address.shape)
address


# ### Encode Category

# In[10]:

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
data["Category(encode)"] = encoder.fit_transform(data["Category"])
      
data.head(3)


# ## Score

# In[11]:

exclude_columns = ["Address", "Address_Type", "Address(encode)", "Category"]

label_name = "Category(encode)"
feature_names = data.columns.difference([label_name] + exclude_columns)

X = data[feature_names]

print(X.shape)
X.head(3)


# In[12]:

from scipy.sparse import hstack

X = hstack((X.values.astype(np.float32), address.astype(np.float32)))
X


# In[13]:

y = data[label_name]

print(y.shape)
y.head(3)


# ### Evaluate using XGBoost

# In[14]:

import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold, train_test_split, cross_val_score

random_state = 37

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=random_state)

watchlist = [(X_train, y_train), (X_valid, y_valid)]

kfold = StratifiedKFold(y_train, n_folds=5, random_state=random_state)

model = xgb.XGBClassifier(
    objective='multi:softprob',
    learning_rate=1.0,
    max_depth=6,
    max_delta_step=1,
    n_estimators=15,
)

fit_params={
    "eval_set": watchlist,
    "eval_metric": "mlogloss",
    "verbose": False,
}

get_ipython().magic("time score = cross_val_score(model, X_train, y_train,                               scoring='log_loss', cv=kfold,                               fit_params=fit_params).mean()")

score = -1.0 * score
print("Use XGBClassifier. Score = {0:.6f}".format(score))


# ## Result
#   * Before = 2.490879
#   * After = 2.298930

# In[ ]:



