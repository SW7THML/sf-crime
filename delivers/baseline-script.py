
# coding: utf-8

# # Baseline script of San Francisco Crime Classification
# 
# Baseline script. Hope this helps.

# In[1]:

import numpy as np
import pandas as pd


# ## Load datasets

# In[2]:

train = pd.read_csv("../data/train.csv", parse_dates=["Dates"])

train.drop("Resolution", axis=1, inplace=True)
train.drop("Descript", axis=1, inplace=True)

print(train.shape)
train.head(3)


# In[3]:

test = pd.read_csv("../data/test.csv", parse_dates=["Dates"])

print(test.shape)
test.head(3)


# In[4]:

sample = pd.read_csv("../data/sampleSubmission.csv", index_col="Id")

print(sample.shape)
sample.head(3)


# ## Feature Engineering

# In[5]:

combi = pd.concat([train, test])

print(combi.shape)
combi.head(3)


# ### DayOfWeek

# In[6]:

print(combi["DayOfWeek"].unique())

day_of_week_dataframe = pd.get_dummies(combi["DayOfWeek"], prefix="DayOfWeek").astype(np.bool)
print(day_of_week_dataframe.shape)
day_of_week_dataframe.head(3)


# In[7]:

combi = pd.concat([combi, day_of_week_dataframe], axis=1)
combi.drop("DayOfWeek", axis=1, inplace=True)

print(combi.shape)
combi.head(3)


# ### PdDistrict

# In[8]:

print(combi["PdDistrict"].unique())

pd_district_dataframe = pd.get_dummies(combi["PdDistrict"], prefix="PdDistrict").astype(np.bool)

print(pd_district_dataframe.shape)
pd_district_dataframe.head(3)


# In[9]:

combi = pd.concat([combi, pd_district_dataframe], axis=1)
combi.drop("PdDistrict", axis=1, inplace=True)

print(combi.shape)
combi.head(3)


# ### Dates

# In[10]:

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


# In[11]:

combi["Dates_year"] = combi["Dates"].dt.year
combi["Dates_month"] = combi["Dates"].dt.month
combi["Dates_day"] = combi["Dates"].dt.day
combi["Dates_hour"] = combi["Dates"].dt.hour
combi["Dates_minute"] = combi["Dates"].dt.minute
combi["Dates_second"] = combi["Dates"].dt.second
combi["Awake"] = combi["Dates_hour"].apply(lambda x: 1 if (x==0 or (x>=8 and x<=23)) else 0)
combi["Summer"], combi["Fall"], combi["Winter"], combi["Spring"]=zip(*combi["Dates_month"].apply(get_season))

combi.drop("Dates", axis=1, inplace=True)

print(combi.shape)
combi.head(3)


# ### One hot encode address

# In[12]:

# 누적값이 200개 이하인 경우는 'Others'로 바꾼다.
address_counts = combi["Address"].value_counts()
other_index = address_counts[address_counts < 200].index
combi.loc[combi["Address"].isin(other_index), "Address"] = "Others"

print("The number of address types = {address}".format(address=len(combi["Address"].value_counts())))
print(combi.shape)
combi.head()


# In[13]:

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(dtype=np.bool)

combi["Address(encode)"] = label_encoder.fit_transform(combi["Address"])
address = one_hot_encoder.fit_transform(combi[["Address(encode)"]])

print(address.shape)
address


# In[14]:

train_address = address[:len(train), :]
test_address = address[len(train):, :]

print("Train = {0}".format(train_address.shape))
print("Test = {0}".format(test_address.shape))


# ### Split to train / test dataset

# In[15]:

combi.drop("Address", axis=1, inplace=True)

print(combi.shape)
combi.head(3)


# In[16]:

train = combi[combi["Category"].notnull()]

train.drop("Id", axis=1, inplace=True)

print(train.shape)
train.head(3)


# In[17]:

test = combi[combi["Category"].isnull()]

test["Id"] = test["Id"].astype(np.int32)
test.drop("Category", axis=1, inplace=True)

test.set_index("Id", inplace=True)

print(test.shape)
test.head(3)


# ## Score

# In[18]:

label_name = "Category"
feature_names = train.columns.difference([label_name])

X_train = train[feature_names]

print(X_train.shape)
X_train.head(3)


# In[19]:

from scipy.sparse import hstack

X_train = hstack((X_train.values.astype(np.float32), train_address.astype(np.float32)))
X_train


# In[20]:

y_train = train[label_name]

print(y_train.shape)
y_train.head(3)


# ### Evaluate using Naive Bayes

# In[21]:

from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score, StratifiedKFold

kfold = StratifiedKFold(y_train, n_folds=6)

model = BernoulliNB()
get_ipython().magic("time score = cross_val_score(model, X_train, y_train, cv=kfold, scoring='log_loss').mean()")
score = -1.0 * score

print("Use BernoulliNB. Score = {0:.6f}".format(score))


# ## Predict

# In[22]:

X_test = test[feature_names]

print(X_test.shape)
X_test.head(3)


# In[23]:

X_test = hstack((X_test.values.astype(np.float32), test_address.astype(np.float32)))
X_test


# In[24]:

from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import cross_val_score, StratifiedKFold

model = BernoulliNB()
model.fit(X_train, y_train)

prediction = model.predict_proba(X_test)

print(prediction.shape)
prediction[:1]


# In[25]:

submission = pd.DataFrame(prediction, index=test.index, columns = sample.columns)
submission = submission.reindex_axis(sorted(submission.columns), axis=1,)

print(submission.shape)
submission.head(3)


# In[26]:

from datetime import datetime

current_time = datetime.now()
current_time = current_time.strftime("%Y%m%d%H%M%S")

description = "one-hot-encode-address"
csv_filename = "../submissions/" + current_time + "_" + description + ".csv"

submission.to_csv(csv_filename)


# In[27]:

import gzip

gzip_filename = csv_filename + ".gz"

f_in = open(csv_filename, "rb")

f_out = gzip.open(gzip_filename, 'wb')
f_out.writelines(f_in)
f_out.close()

f_in.close()


# In[ ]:



