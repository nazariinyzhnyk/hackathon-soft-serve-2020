from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.metrics import fbeta_score

pd.options.display.max_colwidth=100
pd.options.display.max_columns=300

employees = pd.read_csv("data/employees.csv")
history = pd.read_csv("data/history.csv")
submission = pd.read_csv("data/submission.csv")

# history.loc[:,'Date'] = list(pd.to_datetime(history['Date']))

def get_month(text):
    if type(text) == str:
        numbers = text.split("/")
        return int(numbers[0])

def get_year(text):
    if type(text) == str:
        numbers = text.split("/")
        return int(numbers[-1])

df = history.merge(employees)

# for data labeling
def label_df(data, n_month=3):
    labels = []
    for emp in data.EmployeeID.unique():
        curr_emp = list(data[data.EmployeeID == emp]['DismissalDate'])
        len_emp = len(curr_emp)
        if pd.isnull(curr_emp[0]):
            labels += [0 for _ in range(len_emp - n_month)] + [2 for _ in range(n_month)]
        else:
            labels += [0 for _ in range(len_emp - n_month)] + [1 for _ in range(n_month)]
    return labels

lbls = label_df(df)

X_test = df[df.target==2]
X_test = X_test[X_test.Date == datetime(2019, 2, 1)]
X_test = X_test[X_test.EmployeeID.isin(set(submission.EmployeeID))]

df['target'] = lbls
df = df[df.target!=2]


columns_to_drop = ['EmployeeID', 'HiringDate', 'DismissalDate']

cat_columns = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel',
               'IsTrainee', 'LanguageLevelID', 'CustomerID', 'ProjectID',
               'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID',
               'PaymentTypeId']

X = df.drop(columns_to_drop, axis = 1)

from category_encoders.basen import BaseNEncoder
encoder = BaseNEncoder(cols = cat_columns, base = 2)
X = encoder.fit_transform(X)

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random

RANDOM_SEED = 42

def set_seed(random_state=RANDOM_SEED):
    random.seed(random_state)
    np.random.seed(random_state)

set_seed()

def model_fit(X_train, y_train):
    clf2 = RandomForestClassifier(n_estimators=200,
                               min_samples_split=20,
                               min_samples_leaf=20,
                               max_features='sqrt',
                               max_depth=40,
                               bootstrap =True,
                               n_jobs = 8,
                                  class_weight='balanced_subsample')
    # clf2 = LogisticRegression()
    clf2 = clf2.fit(X_train, y_train)
    return clf2

def model_predict(X, models):
    preds = models.predict(X)
    return preds

X["month"] = X["Date"].apply(get_month)
X["year"] = X["Date"].apply(get_year)
X = X.sort_values(["year", "month"])
X["year_month"] = X.apply(lambda row: str(row["year"]) + "_" + str(row["month"]), axis=1)
year_month = list(X.year_month.unique())
mapping_year_month = dict(zip(year_month, range(len(year_month))))
X["order_in_time"] = X["year_month"].map(mapping_year_month)
splits = [[[0, 9], [10, 11]],
          [[0, 11], [12, 13]],
          [[0, 13], [14, 15]],
          [[0, 15], [16, 17]],
          [[0, 17], [18, 19]]]

fold = 0
scores = []
for split in splits:
    fold += 1

    X_train = X.query(f"order_in_time >= " + str(split[0][0]) + " & order_in_time <" + str(split[0][1]))
    X_val = X.query(f"order_in_time >= " + str(split[1][0]) + " & order_in_time <" + str(split[1][1]))

    y_train = X_train['target']
    y_val = list(X_val['target'])

    X_train = X_train.drop(columns=["month", "year", "year_month", "order_in_time", "Date", 'target'])
    X_val = X_val.drop(columns=["month", "year", "year_month", "order_in_time", "Date", 'target'])

    print("fold: ", fold)
    print("train:", len(X_train))
    print("val:", len(X_val))
    print(" ")

    print('FOLD #{}'.format(fold))
    models = model_fit(X_train, y_train)
    print('END OF MODEL FIT')

    y_pred = model_predict(X_val, models)

    score = fbeta_score(y_val, y_pred, beta=1.7)

    print('Validation Score: {}'.format(score))
    scores.append(score)

mean_score = np.mean(scores)
print('MEAN OF SCOREs: {}'.format(mean_score))


