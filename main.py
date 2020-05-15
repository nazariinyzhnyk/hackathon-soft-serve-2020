from datetime import datetime
import os

from sklearn.model_selection import KFold
from sklearn.metrics import fbeta_score
from sklearn.preprocessing import MinMaxScaler
from category_encoders.basen import BaseNEncoder
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

cat_columns = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel',
               'IsTrainee', 'LanguageLevelID', 'CustomerID', 'ProjectID',
               'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID',
               'PaymentTypeId']

submission = pd.read_csv(os.path.join("data", "submission.csv"))

if USE_CACHE:
    df = pd.read_csv('df.csv')
else:
    employees = pd.read_csv(os.path.join("data", "employees.csv"))
    history = pd.read_csv(os.path.join("data", "history.csv"))

    df = history.merge(employees)
    labels = label_df(df)
    df['target'] = labels

    if TEST_MODE:
        df = df.head(1000)

    num_features = ['Utilization', 'HourVacation', 'HourMobileReserve', 'HourLockedReserve', 'BonusOneTime', 'APM',
                    'WageGross']
    cat_features_num = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel', 'IsTrainee', 'LanguageLevelID',
                        'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID', 'PaymentTypeId',
                        'MonthOnPosition', 'MonthOnSalary']
    cat_features_txt = ['CustomerID', 'ProjectID']

    for num_feature in tqdm(num_features):
        df[num_feature + '_mean'] = get_mean_value(df, num_feature)
        df[num_feature + '_std'] = get_std_value(df, num_feature)
        df[num_feature + '_min'] = get_min_value(df, num_feature)
        df[num_feature + '_max'] = get_max_value(df, num_feature)
        df[num_feature + '_nunique'] = get_nunique_values(df, num_feature)
        df[num_feature + '_nunique_frac'] = get_nunique_values_fraction(df, num_feature)
        df[num_feature + '_last_m_max'] = get_last_minus_max(df, num_feature)
        df[num_feature + '_last_m_min'] = get_last_minus_min(df, num_feature)
        df[num_feature + '_last_m_mean'] = get_last_minus_mean(df, num_feature)
        df[num_feature + '_time_since_lat_change'] = get_time_since_last_change(df, num_feature)
    df.to_csv('df.csv', index=False)

    for cat_feature in tqdm(cat_features_num):
        df[cat_feature + '_std'] = get_std_value(df, cat_feature)
        df[cat_feature + '_min'] = get_min_value(df, cat_feature)
        df[cat_feature + '_max'] = get_max_value(df, cat_feature)
        df[cat_feature + '_nunique'] = get_nunique_values(df, cat_feature)
        df[cat_feature + '_nunique_frac'] = get_nunique_values_fraction(df, cat_feature)
        df[cat_feature + '_time_since_lat_change'] = get_time_since_last_change(df, cat_feature)
    df.to_csv('df.csv', index=False)

    # for cat_feature in tqdm(cat_features_txt):
    #     df[cat_feature + '_nunique'] = get_nunique_values(df, cat_feature)
    #     df[cat_feature + '_nunique_frac'] = get_nunique_values_fraction(df, cat_feature)
    #     df[cat_feature + '_time_since_lat_change'] = get_time_since_last_change(df, cat_feature)

    df['Date'] = list(pd.to_datetime(df['Date']))
    df["HiringDate"] = list(pd.to_datetime(df['HiringDate']))
    df["Month_in_copmany"] = df.apply(lambda row: (row['Date'].year - row['HiringDate'].year)*12 + (row['Date'].month -row['HiringDate'].month), axis=1)

    df["month_of_record"] = list(pd.DatetimeIndex(df['Date']).month)

    df["Month_on_employee"] = get_count_month_for_columns(df, "CustomerID")
    df["ProjectID"] = df["ProjectID"].astype(str)
    df["Month_on_project"] = get_count_month_for_columns(df, "ProjectID")

    df["Count_change_salary"] = count_change_salary(df)
    df["Count_month_with_bonuses"] = count_month_with_having_something(df, "HourVacation")
    df["Count_month_with_vacantion"] = count_month_with_having_something(df, "BonusOneTime")

    df.to_csv('df.csv', index=False)
print('Features extracted.')

X_test = df[df.target == 2]
df = df[df.target != 2]

X = df.drop(columns_to_drop, axis=1)

encoder = BaseNEncoder(cols=cat_columns, base=3)
X = encoder.fit_transform(X)


# X["month"] = X["Date"].apply(get_month)
# X["year"] = X["Date"].apply(get_year)
# X = X.sort_values(["year", "month"])

fold = 0
scores = []

kf = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
kf.get_n_splits(X)

y = X.target
X = X.drop(columns=["Date", 'target'])

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


X_test.loc[:, 'Date'] = list(pd.to_datetime(X_test['Date']))
X_test = X_test[X_test.Date == datetime(2019, 2, 1)]
X_test = X_test[X_test.EmployeeID.isin(set(submission.EmployeeID))]
emp_ids = X_test.EmployeeID
X_test = X_test.drop(columns_to_drop, axis=1)
X_test = encoder.transform(X_test)
X_test = X_test.drop(columns=["Date", 'target'])
preds = model.predict(X_test)

result = pd.DataFrame({'EmployeeID': emp_ids, 'target': preds})
result.to_csv('submission.csv', index=False)
