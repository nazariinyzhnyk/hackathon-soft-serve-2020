from datetime import datetime
import os

from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import MinMaxScaler
from category_encoders.basen import BaseNEncoder
from imblearn.ensemble import BalancedRandomForestClassifier

from utils import *
from features import *

RANDOM_SEED = 42
set_seed(RANDOM_SEED)

USE_SCALER = True
USE_CACHED = False
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

columns_to_drop = ['EmployeeID', 'HiringDate', 'DismissalDate']

cat_columns = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel',
               'IsTrainee', 'LanguageLevelID', 'CustomerID', 'ProjectID',
               'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID',
               'PaymentTypeId']

num_features = ['Utilization', 'HourVacation', 'HourMobileReserve', 'HourLockedReserve', 'BonusOneTime', 'APM',
                'WageGross']
cat_features_num = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel', 'IsTrainee', 'LanguageLevelID',
                    'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID', 'PaymentTypeId',
                    'MonthOnPosition', 'MonthOnSalary']

if USE_CACHED:
    df = pd.read_csv('df.csv')
else:
    for num_feature in tqdm(num_features):
        df[num_feature + '_diff'], df[num_feature + '_lag'] = get_prev_values_stats(df, num_feature)
        df[num_feature + '_nuniques'], df[num_feature + '_nunique_fracs'], df[
            num_feature + '_time_since_last_change_vals'] = get_all_feat_values(df, num_feature)
        df[num_feature + '_max'], df[num_feature + '_min'], df[num_feature + '_std'], df[num_feature + '_mean'], \
            df[num_feature + '_lmmin'], df[num_feature + '_lmmax'], \
            df[num_feature + '_lmmean'] = get_num_feat_values(df, num_feature)
    df.to_csv('df.csv', index=False)

    for cat_feature in tqdm(cat_features_num):
        df[cat_feature + '_diff'], df[cat_feature + '_lag'] = get_prev_values_stats(df, cat_feature)
        df[cat_feature + '_nuniques'], df[cat_feature + '_nunique_fracs'], df[
            '_time_since_last_change_vals'] = get_all_feat_values(df, cat_feature)
    df.to_csv('df.csv', index=False)

df_x = df[df.target != 2]
X = df_x.drop(columns_to_drop, axis=1)

encoder = BaseNEncoder(cols=cat_columns, base=2)
X = encoder.fit_transform(X)


X["month"] = X["Date"].apply(get_month)
X["year"] = X["Date"].apply(get_year)
X = X.sort_values(["year", "month"])

fold = 0
scores = []

kf = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
kf.get_n_splits(X)

y = X.target
X = X.drop(columns=["month", "year", "Date", 'target'])

for train_index, test_index in kf.split(X):
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


X_test = df[df.target == 2]
X_test.loc[:, 'Date'] = list(pd.to_datetime(X_test['Date']))

X_test = X_test[X_test.Date == datetime(2019, 2, 1)]
X_test = X_test[X_test.EmployeeID.isin(set(submission.EmployeeID))]
emp_ids = X_test.EmployeeID
X_test = X_test.drop(columns_to_drop, axis=1)
X_test = encoder.transform(X_test)
X_test = X_test.drop(columns=["Date", 'target'])
preds = model.predict(X_test)
print(sum(preds))
result = pd.DataFrame({'EmployeeID': emp_ids, 'target': preds})
result.to_csv('submission.csv', index=False)
