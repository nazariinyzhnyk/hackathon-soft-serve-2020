import os

import pandas as pd
import numpy as np


DATA_DIR = os.path.join('data')
print(os.listdir(DATA_DIR))


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


hist = pd.read_csv(os.path.join(DATA_DIR, 'history.csv'))
empls = pd.read_csv(os.path.join(DATA_DIR, 'employees.csv'))
sbmsn = pd.read_csv(os.path.join(DATA_DIR, 'submission.csv'))

df = hist.merge(empls)
lbls = label_df(df)

unique, counts = np.unique(lbls, return_counts=True)
print(dict(zip(unique, counts)))

df['target'] = lbls
print(df.head(60))
