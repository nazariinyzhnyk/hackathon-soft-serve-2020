import numpy as np
from scipy.ndimage.interpolation import shift


def get_prev_values_stats(df, column):
    employees = df.EmployeeID.unique()
    diff_values_emps = []
    lag_values_emps = []
    for empl in employees:
        column_values = np.array(df[df.EmployeeID == empl][column].values)
        diffs = np.insert(np.diff(column_values), 0, 0)
        diff_values_emps += list(diffs)
        lag_values_emps += list(shift(column_values, 1, cval=np.nan))
    return diff_values_emps, lag_values_emps

def get_num_feat_values(df, column):
    employees = df.EmployeeID.unique()
    maxs = []
    mins = []
    stds = []
    means = []
    last_minus_mins = []
    last_minus_maxs = []
    last_minus_means = []
    for empl in employees:
        column_values = np.array(df[df.EmployeeID == empl][column].values)
        for i in range(len(column_values)):
            if i == 0:
                maxs.append(0)
                mins.append(0)
                stds.append(0)
                means.append(0)
                last_minus_mins.append(0)
                last_minus_maxs.append(0)
                last_minus_means.append(0)
            else:
                sub_list = column_values[:i]
                last_val = sub_list[len(sub_list) - 1]
                maxx = np.max(sub_list)
                minn = np.min(sub_list)
                stdd = np.std(sub_list)
                avgg = np.mean(sub_list)
                maxs.append(maxx)
                mins.append(minn)
                stds.append(stdd)
                means.append(avgg)
                last_minus_mins.append(last_val - maxx)
                last_minus_maxs.append(last_val - minn)
                last_minus_means.append(last_val - avgg)
    return maxs, mins, stds, means, last_minus_mins, last_minus_maxs, last_minus_means


def get_all_feat_values(df, column):
    employees = df.EmployeeID.unique()
    nuniques = []
    nunique_fracs = []
    time_since_last_change_vals = []
    for empl in employees:
        column_values = np.array(df[df.EmployeeID == empl][column].values)
        for i in range(len(column_values)):
            if i == 0:
                nuniques.append(0)
                nunique_fracs.append(0)
                time_since_last_change_vals.append(0)
            else:
                sub_list = column_values[:i]
                n_unique = len(np.unique(sub_list))
                nuniques.append(n_unique)
                nunique_fracs.append(n_unique / len(sub_list))
                not_eq = list(np.argwhere(np.array(sub_list) != sub_list[-1]))
                if not_eq:
                    count_changes = np.argwhere(np.array(sub_list) != sub_list[-1])[0][0]
                else:
                    count_changes = 0
                time_since_last_change_vals.append(count_changes)
    return nuniques, nunique_fracs, time_since_last_change_vals

def model_fit(x_train, y_train, classifier, **params):
    clf2 = classifier(**params)
    clf2 = clf2.fit(x_train, y_train)
    return clf2


def model_predict(data, model):
    return model.predict(data)

import pandas as pd
import numpy as np

from sklearn.metrics import fbeta_score

pd.options.display.max_colwidth=100
pd.options.display.max_columns=300

employees = pd.read_csv("data/employees.csv")
history = pd.read_csv("data/history.csv")
submission = pd.read_csv("data/submission.csv")

history.loc[:,'Date'] = list(pd.to_datetime(history['Date']))

def get_month(text):
    if type(text) == str:
        numbers = text.split("/")
        return int(numbers[0])

def get_year(text):
    if type(text) == str:
        numbers = text.split("/")
        return int(numbers[-1])

df = history.merge(employees)
# df = df.head(1002)  # ##################### DELTETE THIS


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
print(len(df))
print(len(lbls))
df['target'] = lbls

from tqdm import tqdm
USE_CACHED = False

####### Nazarii part
columns_to_drop = ['EmployeeID', 'HiringDate', 'DismissalDate']

cat_columns = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel',
               'IsTrainee', 'LanguageLevelID', 'CustomerID', 'ProjectID',
               'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID',
               'PaymentTypeId']

num_features = ['Utilization', 'HourVacation', 'BonusOneTime', 'APM',
                'WageGross', 'MonthOnPosition', 'MonthOnSalary']

cat_features_num = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel', 'IsTrainee', 'LanguageLevelID',
                    'IsInternalProject', 'OnSite', 'CompetenceGroupID', 'FunctionalOfficeID', 'PaymentTypeId']

if USE_CACHED:
    df = pd.read_csv('df.csv')
else:
    for cat_feature in tqdm(cat_features_num):
        _ , df[cat_feature + '_lag'] = get_prev_values_stats(df, cat_feature)
        df[cat_feature + '_nuniques'], df[cat_feature + '_nunique_fracs'], df[
            '_time_since_last_change_vals'] = get_all_feat_values(df, cat_feature)
    df.to_csv('df.csv', index=False)

df['HourLockedReserve'] = (df['HourLockedReserve']>0).astype(int)
df['HourMobileReserve'] = (df['HourMobileReserve']>0).astype(int)
# df['Years_in_company'] = ((df.Date - pd.to_datetime(df['HiringDate'])).values/3.154e+16).astype(int)

# WageGross
df.loc[list((df[df.WageGross<0]).index), 'WageGross'] = 0
df.loc[list((df[df.WageGross>10]).index), 'WageGross'] = 10

# BonusOneTime
df.loc[list((df[df.WageGross<0]).index), 'BonusOneTime'] = 0
df.loc[list((df[df.WageGross>3000]).index), 'BonusOneTime'] = 3000

# APM
df.loc[list((df[df.APM<0]).index), 'APM'] = 0
df.loc[list((df[df.APM>100]).index), 'APM'] = 100

# binning LanguageLevelID
cut_labels_4 = ['low', 'low/medium', 'medium/high', 'high']
cut_bins = [0, 6, 12, 18, 27]
df['LanguageLevelID'] = pd.cut(df['LanguageLevelID'], bins=cut_bins, labels=cut_labels_4)
df['LanguageLevelID_lag'] = pd.cut(df['LanguageLevelID_lag'], bins=cut_bins, labels=cut_labels_4)

# binning PositionLevel
cut_labels_4 = ['low', 'low/medium', 'medium/high', 'high']
cut_bins = [0, 2, 5, 8, 11]
df['PositionLevel'] = pd.cut(df['PositionLevel'], bins=cut_bins, labels=cut_labels_4)
df['PositionLevel_lag'] = pd.cut(df['PositionLevel_lag'], bins=cut_bins, labels=cut_labels_4)

# process DevCenterID
tmp = df.DevCenterID.value_counts()
new_client_id = {}
for row in tmp.iteritems():
    if row[1]>1000:
        new_client_id[row[0]] = row[0]
    else:
        new_client_id[row[0]] = -999

new_cust_id = [new_client_id.get(i) for i in df.DevCenterID]
df.DevCenterID = new_cust_id
new_cust_id = [new_client_id.get(i) for i in df.DevCenterID_lag]
df.DevCenterID_lag = new_cust_id

# process ProjectID
tmp = df.ProjectID.value_counts()
new_client_id = {}
for row in tmp.iteritems():
    if row[1]>70:
        new_client_id[row[0]] = row[0]
    else:
        new_client_id[row[0]] = -999

new_cust_id = [new_client_id.get(i) for i in df.ProjectID]
df.ProjectID = new_cust_id

# process CustomerID
tmp = df.CustomerID.value_counts()
new_client_id = {}
for row in tmp.iteritems():
    if row[1]>50:
        new_client_id[row[0]] = row[0]
    else:
        new_client_id[row[0]] = -999

new_cust_id = [new_client_id.get(i) for i in df.CustomerID]
df.CustomerID = new_cust_id

# process PositionID
tmp = df.PositionID.value_counts()
new_client_id = {}
for row in tmp.iteritems():
    if row[1]>100:
        new_client_id[row[0]] = row[0]
    else:
        new_client_id[row[0]] = -999

new_cust_id = [new_client_id.get(i) for i in df.PositionID]
df.PositionID = new_cust_id
new_cust_id = [new_client_id.get(i) for i in df.PositionID_lag]
df.PositionID_lag = new_cust_id

# process SBUID
tmp = df.SBUID.value_counts()
new_client_id = {}
for row in tmp.iteritems():
    if row[1]>100:
        new_client_id[row[0]] = row[0]
    else:
        new_client_id[row[0]] = -999

new_cust_id = [new_client_id.get(i) for i in df.SBUID]
df.SBUID = new_cust_id
new_cust_id = [new_client_id.get(i) for i in df.SBUID_lag]
df.SBUID_lag = new_cust_id

# process PaymentTypeId
tmp = df.PaymentTypeId.value_counts()
new_client_id = {}
for row in tmp.iteritems():
    if row[1]>1000:
        new_client_id[row[0]] = row[0]
    else:
        new_client_id[row[0]] = -999

new_cust_id = [new_client_id.get(i) for i in df.PaymentTypeId]
df.PaymentTypeId = new_cust_id
new_cust_id = [new_client_id.get(i) for i in df.PaymentTypeId_lag]
df.PaymentTypeId_lag = new_cust_id

# process FunctionalOfficeID
tmp = df.FunctionalOfficeID.value_counts()
new_client_id = {}
for row in tmp.iteritems():
    if row[1]>10000:
        new_client_id[row[0]] = row[0]
    else:
        new_client_id[row[0]] = -999

new_cust_id = [new_client_id.get(i) for i in df.FunctionalOfficeID]
df.FunctionalOfficeID = new_cust_id
new_cust_id = [new_client_id.get(i) for i in df.FunctionalOfficeID_lag]
df.FunctionalOfficeID_lag = new_cust_id

# process CompetenceGroupID
tmp = df.CompetenceGroupID.value_counts()
new_client_id = {}
for row in tmp.iteritems():
    if row[1]>500:
        new_client_id[row[0]] = row[0]
    else:
        new_client_id[row[0]] = -999

new_cust_id = [new_client_id.get(i) for i in df.CompetenceGroupID]
df.CompetenceGroupID = new_cust_id
new_cust_id = [new_client_id.get(i) for i in df.CompetenceGroupID_lag]
df.CompetenceGroupID_lag = new_cust_id

### Nazarii block
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

df = pd.read_csv('df.csv')
print(f'nrows before deletion: {len(df)}')
df = df[df['WageGross_lag'].notna()]
print(f'nrows after deletion: {len(df)}')

# dropping unpropriate columns:
columns_to_drop = ['EmployeeID', 'HiringDate', 'DismissalDate']
X = df.drop(columns_to_drop, axis = 1)

cat_features_num = ['PositionLevel', 'IsTrainee', 'LanguageLevelID',
                    'IsInternalProject', 'OnSite', 'FunctionalOfficeID']

dummie_features = ['FunctionalOfficeID','PositionLevel', 'IsTrainee',
                   'LanguageLevelID', 'IsInternalProject', 'HourLockedReserve',
                   'HourMobileReserve', 'OnSite'] + [i + '_lag' for i in cat_features_num]

binary_features = ['DevCenterID', 'PaymentTypeId', 'CompetenceGroupID', 'CustomerID',
                   'DevCenterID_lag', 'PaymentTypeId_lag', 'CompetenceGroupID_lag']

triple_features = ['SBUID', 'PositionID', 'ProjectID', 'SBUID_lag', 'PositionID_lag']

numeric = ['Utilization', 'HourVacation', 'BonusOneTime', 'APM',
           'WageGross', 'MonthOnPosition', 'MonthOnSalary']


# encodding features:
from category_encoders.basen import BaseNEncoder
encoder_dummie = BaseNEncoder(cols = dummie_features, base = 1)
encoder_binary = BaseNEncoder(cols = binary_features, base = 2)
encoder_3_huli = BaseNEncoder(cols = triple_features, base = 3)

X = encoder_dummie.fit_transform(X)
X = encoder_binary.fit_transform(X)
X = encoder_3_huli.fit_transform(X)

# !pip install imblearn
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=100)


def model_fit(X_train, y_train):
    clf2 = BalancedRandomForestClassifier(n_estimators=200, n_jobs = 8)
    clf2 = clf2.fit(X_train, y_train)
    return clf2

def model_predict(X, models):
    preds = models.predict(X)
    return preds


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

USE_SCALER = True
RANDOM_SEED = 42
fold = 0
scores = []

kf = KFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
kf.get_n_splits(X)

tmp = 0
tmp2 = 0

X_test = X[(df['target'] == 2) & (df.EmployeeID.isin(set(submission.EmployeeID)))]
X = X[X.target != 2]
y = X.target.values

X = X.drop(columns=["Date", 'target'])
X_test = X_test.drop(columns=["Date", 'target'])

from sklearn.preprocessing import StandardScaler
sclr = StandardScaler()
X = sclr.fit_transform(X)

X = pca.fit_transform(X)
X_test = sclr.transform(X_test)
X_test = pca.transform(X_test)

# for train_index, test_index in kf.split(X):
#     fold += 1
#     X_train, X_val = X[train_index], X[test_index]
#     y_train, y_val = y[train_index], y[test_index]

#     if USE_SCALER:
#         X_train = scaler.fit_transform(X_train)
#         X_val = scaler.transform(X_val)

#     print("fold: ", fold)
#     print("train:", len(X_train))
#     print("val:",len(X_val))
#     print(" ")

#     print('FOLD #{}'.format(fold))
#     models = model_fit(X_train, y_train)
#     print('END OF MODEL FIT')

#     y_pred = model_predict(X_val, models)

#     score = fbeta_score(y_val, y_pred, beta=1.7)

#     print('Validation Score: {}'.format(score))
#     scores.append(score)

# mean_score = np.mean(scores)
# print('MEAN OF SCOREs: {}'.format(mean_score))

model = model_fit(X, y)

from datetime import datetime

emp_ids = df[(df['target'] == 2)&(df.EmployeeID.isin(set(submission.EmployeeID)))]['EmployeeID']

# X_test = X_test.drop(columns=['EmployeeID',"Date", 'target'])
preds = model.predict(X_test)
pred_co = []
for i in range(0, len(preds), 3):
    if sum(preds[i:i+3]) >= 2:
        pred_co.append(1)
    else:
        pred_co.append(0)

print(sum(preds))
result = pd.DataFrame({'EmployeeID': emp_ids.drop_duplicates(), 'target': pred_co})

result.to_csv('submission_NAZARII_FEATURES_retried.csv', index=False)

print(sum(pred_co))
