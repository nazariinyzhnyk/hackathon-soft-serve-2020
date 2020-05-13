from datetime import datetime
import os

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import MinMaxScaler
from category_encoders.basen import BaseNEncoder
from imblearn.ensemble import BalancedRandomForestClassifier

from utils import *

RANDOM_SEED = 42
set_seed(RANDOM_SEED)

USE_SCALER = True
scaler = MinMaxScaler()

cls = BalancedRandomForestClassifier
cls_params = {
    "n_estimators": 200,
    "n_jobs": 8
}

employees = pd.read_csv(os.path.join("data", "employees.csv"))
history = pd.read_csv(os.path.join("data", "history.csv"))
submission = pd.read_csv(os.path.join("data", "submission.csv"))

df = history.merge(employees)
labels = label_df(df)
df['target'] = labels
df = df[df.target != 2]

columns_to_drop = ['EmployeeID', 'HiringDate', 'DismissalDate']

cat_columns = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel',
               'IsTrainee', 'LanguageLevelID', 'CustomerID', 'ProjectID',
               'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID',
               'PaymentTypeId']

X = df.drop(columns_to_drop, axis=1)

encoder = BaseNEncoder(cols=cat_columns, base=2)
X = encoder.fit_transform(X)


X["month"] = X["Date"].apply(get_month)
X["year"] = X["Date"].apply(get_year)
X = X.sort_values(["year", "month"])

fold = 0
scores = []

folds = get_cv_folds(data=df, n_fold=2)

# kf = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
# kf.get_n_splits(X)

y = X.target
X = X.drop(columns=["month", "year", "Date", 'target'])

# for train_index, test_index in kf.split(X):
for train_index, test_index in folds:
    fold += 1
    X_train, X_val = X.loc[X.index.intersection(train_index)], X.loc[X.index.intersection(test_index)]
    y_train, y_val = y.loc[y.index.intersection(train_index)], y.loc[y.index.intersection(test_index)]

    if USE_SCALER:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

    print(f'FOLD #{fold}')
    print("train:", len(X_train))
    print("val:", len(X_val))
    model = model_fit(X_train, y_train, classifier=cls, **cls_params)

    y_pred = model_predict(X_val, model)
    score = fbeta_score(y_val, y_pred, beta=1.7)
    print('Validation Score: {}'.format(score))
    scores.append(score)

print(f'MEAN OF SCOREs: {np.mean(scores)}')

model = model_fit(X, y, classifier=cls, **cls_params)


history.loc[:, 'Date'] = list(pd.to_datetime(history['Date']))
df = history.merge(employees)
df['target'] = labels
X_test = df[df.target == 2]
X_test = X_test[X_test.Date == datetime(2019, 2, 1)]
X_test = X_test[X_test.EmployeeID.isin(set(submission.EmployeeID))]
emp_ids = X_test.EmployeeID
X_test = X_test.drop(columns_to_drop, axis=1)
X_test = encoder.transform(X_test)
X_test = X_test.drop(columns=["Date", 'target'])
preds = model.predict(X_test)

result = pd.DataFrame({'EmployeeID': emp_ids, 'target': preds})
result.to_csv('submission.csv', index=False)
