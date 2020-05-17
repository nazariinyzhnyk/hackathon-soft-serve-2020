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
        lag_values_emps += list(shift(column_values, 1, cval=0))
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

