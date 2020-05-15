import copy
import numpy as np


def get_count_month_for_columns(df, column = "CustomerID"):
    employees = df.EmployeeID.unique()
    final_days_on_the_project = []
    for empl in employees:
        project_emp = df[df.EmployeeID == empl][column].values

        count_month = 1
        month_on_project = [1]

        cur_project = project_emp[0]
        for i in range(1, len(project_emp)):
            if project_emp[i] == cur_project:
                count_month += 1
            else:
                count_month = 1
                cur_project = project_emp[i]

            month_on_project.append(copy.deepcopy(count_month))
        final_days_on_the_project.extend(month_on_project)
    return final_days_on_the_project

def count_change_salary(df):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        month_on_salary = df[df.EmployeeID == empl]["MonthOnSalary"]
        count_changes = (month_on_salary == 1).sum()
        one_emp_values = [copy.deepcopy(count_changes)]*len(month_on_salary)
        values_to_write.extend(one_emp_values)
    return values_to_write

def count_month_with_having_something(df, column="HourVacation"):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column]
        count_changes = (column_values != 0).sum()
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_mean_value(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = np.mean(column_values)
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_std_value(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = np.std(column_values)
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_min_value(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = np.min(column_values)
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_max_value(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = np.max(column_values)
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_nunique_values(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = len(np.unique(column_values))
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_nunique_values_fraction(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = len(np.unique(column_values)) / len(column_values)
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_last_minus_max(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = column_values[len(column_values) - 1] - np.max(column_values)
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_last_minus_min(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = column_values[len(column_values) - 1] - np.min(column_values)
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_last_minus_mean(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        count_changes = column_values[len(column_values) - 1] - np.mean(column_values)
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write

def get_time_since_last_change(df, column):
    employees = df.EmployeeID.unique()
    values_to_write = []
    for empl in employees:
        column_values = df[df.EmployeeID == empl][column].values
        not_eq = list(np.argwhere(np.array(column_values) != column_values[-1]))
        if not_eq:
            count_changes = np.argwhere(np.array(column_values) != column_values[-1])[0][0]
        else:
            count_changes = 0
        one_emp_values = [copy.deepcopy(count_changes)]*len(column_values)
        values_to_write.extend(one_emp_values)
    return values_to_write
