from datetime import datetime
import os

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import MinMaxScaler
from category_encoders.basen import BaseNEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from category_encoders.one_hot import OneHotEncoder

from imblearn.ensemble import BalancedRandomForestClassifier
from tqdm import tqdm

from utils import *
from notebooks.features import *

RANDOM_SEED = 42
set_seed(RANDOM_SEED)

USE_SCALER = True
USE_CACHE = True
TEST_MODE = False
scaler = MinMaxScaler()

cls = BalancedRandomForestClassifier
cls_params = {
    "n_estimators": 500,
    "n_jobs": 8
}

columns_to_drop = ['EmployeeID', 'HiringDate', 'DismissalDate']
columns_to_drop2 = ['HiringDate', 'DismissalDate']

cat_columns = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel',
               'IsTrainee', 'LanguageLevelID', 'CustomerID', 'ProjectID',
               'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID',
               'PaymentTypeId']

submission = pd.read_csv(os.path.join("data", "submission.csv"))
num_features = ['Utilization', 'HourVacation', 'HourMobileReserve', 'HourLockedReserve', 'BonusOneTime', 'APM',
                    'WageGross']
cat_features_num = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel', 'IsTrainee', 'LanguageLevelID',
                        'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID', 'PaymentTypeId',
                        'MonthOnPosition', 'MonthOnSalary']
cat_features_txt = ['CustomerID', 'ProjectID']

# if USE_CACHE:
#     df = pd.read_csv('df.csv')
# else:
#     employees = pd.read_csv(os.path.join("data", "employees.csv"))
#     history = pd.read_csv(os.path.join("data", "history.csv"))
#
#     df = history.merge(employees)
#     labels = label_df(df)
#     df['target'] = labels
#
#     if TEST_MODE:
#         df = df.head(1000)
#
#     for num_feature in tqdm(num_features):
#         df[num_feature + '_mean'] = get_mean_value(df, num_feature)
#         df[num_feature + '_std'] = get_std_value(df, num_feature)
#         df[num_feature + '_min'] = get_min_value(df, num_feature)
#         df[num_feature + '_max'] = get_max_value(df, num_feature)
#         df[num_feature + '_nunique'] = get_nunique_values(df, num_feature)
#         df[num_feature + '_nunique_frac'] = get_nunique_values_fraction(df, num_feature)
#         df[num_feature + '_last_m_max'] = get_last_minus_max(df, num_feature)
#         df[num_feature + '_last_m_min'] = get_last_minus_min(df, num_feature)
#         df[num_feature + '_last_m_mean'] = get_last_minus_mean(df, num_feature)
#         df[num_feature + '_time_since_lat_change'] = get_time_since_last_change(df, num_feature)
#     df.to_csv('df.csv', index=False)
#
#     for cat_feature in tqdm(cat_features_num):
#         df[cat_feature + '_std'] = get_std_value(df, cat_feature)
#         df[cat_feature + '_min'] = get_min_value(df, cat_feature)
#         df[cat_feature + '_max'] = get_max_value(df, cat_feature)
#         df[cat_feature + '_nunique'] = get_nunique_values(df, cat_feature)
#         df[cat_feature + '_nunique_frac'] = get_nunique_values_fraction(df, cat_feature)
#         df[cat_feature + '_time_since_lat_change'] = get_time_since_last_change(df, cat_feature)
#     df.to_csv('df.csv', index=False)
#
#     # for cat_feature in tqdm(cat_features_txt):
#     #     df[cat_feature + '_nunique'] = get_nunique_values(df, cat_feature)
#     #     df[cat_feature + '_nunique_frac'] = get_nunique_values_fraction(df, cat_feature)
#     #     df[cat_feature + '_time_since_lat_change'] = get_time_since_last_change(df, cat_feature)
#
#     df['Date'] = list(pd.to_datetime(df['Date']))
#     df["HiringDate"] = list(pd.to_datetime(df['HiringDate']))
#     df["Month_in_copmany"] = df.apply(lambda row: (row['Date'].year - row['HiringDate'].year)*12 + (row['Date'].month -row['HiringDate'].month), axis=1)
#
#     df["month_of_record"] = list(pd.DatetimeIndex(df['Date']).month)
#
#     df["Month_on_employee"] = get_count_month_for_columns(df, "CustomerID")
#     df["ProjectID"] = df["ProjectID"].astype(str)
#     df["Month_on_project"] = get_count_month_for_columns(df, "ProjectID")
#
#     df["Count_change_salary"] = count_change_salary(df)
#     df["Count_month_with_bonuses"] = count_month_with_having_something(df, "HourVacation")
#     df["Count_month_with_vacantion"] = count_month_with_having_something(df, "BonusOneTime")
#
#     df.to_csv('df.csv', index=False)

employees = pd.read_csv(os.path.join("data", "employees.csv"))
history = pd.read_csv(os.path.join("data", "history.csv"))

df = history.merge(employees)


labels = label_df(df)
df['target'] = labels
# df = df.head(1000)
small_df = df[df['EmployeeID'] == df.EmployeeID.unique()[0]]
small_df = small_df.head(len(small_df) - 4)

for emp in df.EmployeeID.unique()[1:]:
    emp_df = df[df['EmployeeID'] == emp]
    small_df = small_df.append(emp_df.head(len(emp_df) - 4))

for num_feature in tqdm(num_features):
    small_df[num_feature + '_mean'] = get_mean_value(small_df, num_feature)
    small_df[num_feature + '_std'] = get_std_value(small_df, num_feature)
    small_df[num_feature + '_min'] = get_min_value(small_df, num_feature)
    small_df[num_feature + '_max'] = get_max_value(small_df, num_feature)
    small_df[num_feature + '_nunique'] = get_nunique_values(small_df, num_feature)
    small_df[num_feature + '_nunique_frac'] = get_nunique_values_fraction(small_df, num_feature)
    small_df[num_feature + '_last_m_max'] = get_last_minus_max(small_df, num_feature)
    small_df[num_feature + '_last_m_min'] = get_last_minus_min(small_df, num_feature)
    small_df[num_feature + '_last_m_mean'] = get_last_minus_mean(small_df, num_feature)
    small_df[num_feature + '_time_since_lat_change'] = get_time_since_last_change(small_df, num_feature)
small_df.to_csv('small_df.csv', index=False)

for cat_feature in tqdm(cat_features_num):
    small_df[cat_feature + '_std'] = get_std_value(small_df, cat_feature)
    small_df[cat_feature + '_min'] = get_min_value(small_df, cat_feature)
    small_df[cat_feature + '_max'] = get_max_value(small_df, cat_feature)
    small_df[cat_feature + '_nunique'] = get_nunique_values(small_df, cat_feature)
    small_df[cat_feature + '_nunique_frac'] = get_nunique_values_fraction(small_df, cat_feature)
    small_df[cat_feature + '_time_since_lat_change'] = get_time_since_last_change(small_df, cat_feature)
small_df.to_csv('small_df.csv', index=False)

small_df['Date'] = list(pd.to_datetime(small_df['Date']))
small_df["HiringDate"] = list(pd.to_datetime(small_df['HiringDate']))
small_df["Month_in_copmany"] = small_df.apply(lambda row: (row['Date'].year - row['HiringDate'].year)*12 + (row['Date'].month -
                                                                                                            row['HiringDate'].month), axis=1)

small_df["month_of_record"] = list(pd.DatetimeIndex(small_df['Date']).month)

small_df["Month_on_employee"] = get_count_month_for_columns(small_df, "CustomerID")
small_df["ProjectID"] = small_df["ProjectID"].astype(str)
small_df["Month_on_project"] = get_count_month_for_columns(small_df, "ProjectID")

small_df["Count_change_salary"] = count_change_salary(small_df)
small_df["Count_month_with_bonuses"] = count_month_with_having_something(small_df, "HourVacation")
small_df["Count_month_with_vacantion"] = count_month_with_having_something(small_df, "BonusOneTime")

small_df.to_csv('small_df.csv', index=False)

print('Features extracted.')

df = pd.read_csv('df.csv')
X_test = df[df.target == 2]
df = df[df.target == 1]
df = df.groupby('EmployeeID').last()
small_df = small_df.groupby('EmployeeID').last()
del df['Date']
del df['HiringDate']
del df['DismissalDate']
del small_df['Date']
del small_df['HiringDate']
del small_df['DismissalDate']

X = df.append(small_df)
# X = df.groupby('EmployeeID').last()
# X = X.drop(columns_to_drop, axis=1)
cols = list(set(list(X_test)).intersection(set(list(X))))
X = X[cols]
X_test = X_test[cols]
encoder = OneHotEncoder(cols=cat_columns)
X = encoder.fit_transform(X)


# X["month"] = X["Date"].apply(get_month)
# X["year"] = X["Date"].apply(get_year)
# X = X.sort_values(["year", "month"])

fold = 0
scores = []

kf = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
kf.get_n_splits(X)

y = X.target
X = X.drop(columns=['target'])
X = X.fillna(X.mean())
# for train_index, test_index in kf.split(X):
#     fold += 1
#     X_train, X_val = X.loc[X.index.intersection(train_index)], X.loc[X.index.intersection(test_index)]
#     y_train, y_val = y.loc[y.index.intersection(train_index)], y.loc[y.index.intersection(test_index)]
#
#     if USE_SCALER:
#         X_train = scaler.fit_transform(X_train)
#         X_val = scaler.transform(X_val)
#
#     print(f'FOLD #{fold}')
#     print("train:", len(X_train))
#     print("val:", len(X_val))
#     model = model_fit(X_train, y_train, classifier=cls, **cls_params)
#
#     y_pred = model_predict(X_val, model)
#     score = fbeta_score(y_val, y_pred, beta=1.7)
#     print('Validation Score: {}'.format(score))
#     scores.append(score)
#
# print(f'MEAN OF SCOREs: {np.mean(scores)}')
#
# model = model_fit(X, y, classifier=cls, **cls_params)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rfc = RandomForestClassifier(random_state=42, n_jobs=8)

# param_grid = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [4, 5, 6, 7, 8],
#     'criterion': ['gini', 'entropy']
# }
param_grid = {
    'n_estimators': [500, 1000],
    'max_features': ['auto'],
    'max_depth': [8, 16, 32],
    'criterion': ['gini']
}

print('starting cv grid search')
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
CV_rfc.fit(x_train, y_train)

best_params = CV_rfc.best_params_
print(best_params)

rfc1 = RandomForestClassifier(random_state=42, n_jobs=8,
                              max_features=best_params['max_features'],
                              n_estimators=best_params['n_estimators'],
                              max_depth=best_params['max_depth'],
                              criterion=best_params['criterion'])
# rfc1 = RandomForestClassifier(random_state=42, n_jobs=8,
#                               max_features='auto',
#                               n_estimators=500,
#                               max_depth=32,
#                               criterion='gini')

rfc1.fit(x_train, y_train)
pred = rfc1.predict(x_test)
print("Accuracy for Random Forest on CV data: ", accuracy_score(y_test, pred))
print("F-beta   for Random Forest on CV data: ", fbeta_score(y_test, pred, beta=1.7))
#
rfc1 = RandomForestClassifier(random_state=42, n_jobs=8,
                              max_features=best_params['max_features'],
                              n_estimators=best_params['n_estimators'],
                              max_depth=best_params['max_depth'],
                              criterion=best_params['criterion'])
# rfc1 = RandomForestClassifier(random_state=42, n_jobs=8,
#                               max_features='auto',
#                               n_estimators=500,
#                               max_depth=32,
#                               criterion='gini')
rfc1.fit(X, y)
df = pd.read_csv('df.csv')
df = df[df.target == 2]
X_test.loc[:, 'Date'] = list(pd.to_datetime(df['Date']))
X_test['EmployeeID'] = df['EmployeeID']
X_test = X_test[X_test.Date == datetime(2019, 2, 1)]
X_test = X_test[X_test.EmployeeID.isin(set(submission.EmployeeID))]
emp_ids = X_test.EmployeeID

X_test = X_test.drop(columns=["Date", 'EmployeeID'])

X_test = encoder.transform(X_test)
# X_test = X_test.drop(columns=["Date", 'target'])
del X_test['target']

preds = rfc1.predict_proba(X_test)
p_sum = [1 if p[1] > 0.5 else 0 for p in preds]
print(f'1 before thr: {sum(p_sum)}')

preds_thr = [1 if p[1] > 0.4 else 0 for p in preds]
print(f'1 before thr: {sum(preds_thr)}')

result = pd.DataFrame({'EmployeeID': emp_ids, 'target': preds_thr})
result.to_csv('submission.csv', index=False)
