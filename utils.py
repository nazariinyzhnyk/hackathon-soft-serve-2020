import random

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


def set_seed(random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)


def model_fit(x_train, y_train, classifier, **params):
    clf2 = classifier(**params)
    clf2 = clf2.fit(x_train, y_train)
    return clf2


def model_predict(data, model):
    return model.predict(data)
