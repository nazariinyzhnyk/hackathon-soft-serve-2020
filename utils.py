import random
import warnings

import pandas as pd
import numpy as np


def get_month(text):
    if type(text) == str:
        numbers = text.split("/")
        return int(numbers[0])


def get_year(text):
    if type(text) == str:
        numbers = text.split("/")
        return int(numbers[-1])


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


def get_cv_folds(data, n_fold=5):
    fold_idx = [[[], []] for _ in range(n_fold)]
    for emp in data.EmployeeID.unique():
        wdf = data[data['EmployeeID'] == emp]
        emp_idx = list(wdf.index)
        len_emp = len(emp_idx)

        if len_emp <= n_fold:
            warnings.warn(f'len_emp ({len_emp}) <= n_fold ({n_fold}). will use last observation as test for all folds.')
            for i in range(n_fold):
                fold_idx[i][0] += emp_idx[:-1]
                fold_idx[i][1] += [emp_idx[-1]]
        else:
            for i in range(n_fold, 0, -1):
                fold_idx[n_fold - i][0] += emp_idx[:-i]
                fold_idx[n_fold - i][1] += [emp_idx[-i]]

    return fold_idx


def get_idx_of_last_records(data):
    data['idx'] = [i for i in range(len(data))]
    return list(data.groupby('EmployeeID').last()['idx'])


def elements_in_list(lst, to_check):
    return [l in to_check for l in lst]


def set_seed(random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)


def model_fit(x_train, y_train, classifier, **params):
    clf2 = classifier(**params)
    clf2 = clf2.fit(x_train, y_train)
    return clf2


def model_predict(data, model):
    return model.predict(data)
