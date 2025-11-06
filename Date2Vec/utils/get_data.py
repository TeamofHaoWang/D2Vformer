import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np


def timefeature(dates):
    # -------------------------------------------------------------
    #   Use 'date' attribute to expand the time dimension
    #   The purpose is for later embedding operations
    #   Input dataset must have a 'date' attribute in the format year/month/day/hour.
    #   If any part is missing, modify the corresponding feature below accordingly.
    # -------------------------------------------------------------

    # ---------------------------------------------
    #   After using pd.to_datetime to handle year/month/day,
    #   the .hour, .day attributes can directly extract corresponding time information
    # ---------------------------------------------

    dates["hour"] = dates["date"].apply(lambda row: row.hour / 23 - 0.5, 1)  # Hour of the day
    dates["weekday"] = dates["date"].apply(lambda row: row.weekday() / 6 - 0.5, 1)  # Day of the week
    dates["day"] = dates["date"].apply(lambda row: row.day / 30 - 0.5, 1)  # Day of the month
    dates["month"] = dates["date"].apply(lambda row: row.month / 365 - 0.5, 1)  # Day of the year
    return dates[["hour", "weekday", "day", "month"]].values


def date_feature(date_raw, df):
    date_day = df['date'].dt.floor('D')
    df['date_day'] = date_day

    extra_mark = []

    for i in range(0, len(df['date'])):
        date_feature = date_raw.loc[df['date_day'][i]]  # Extract the Series for each date
        data_mark = np.array(date_feature)
        extra_mark.append(data_mark)

    extra_mark = np.array(extra_mark)
    return extra_mark


def get_data(path, date_path, **kwargs):
    date_features = ['date', 'abs_days', 'year', 'day', 'year_day', 'week', 'lunar_year',
                     'lunar_month', 'lunar_day', 'lunar_year_day', 'dayofyear', 'dayofmonth',
                     'monthofyear', 'dayofweek', 'dayoflunaryear', 'dayoflunarmonth',
                     'monthoflunaryear', 'holidays', 'workdays', 'residual_holiday',
                     'residual_workday', 'jieqiofyear', 'jieqi_day', 'dayofjieqi']

    if isinstance(date_path, list):
        date_raw_list = []
        for i in range(len(date_path)):
            date_raw_list.append(pd.read_csv(date_path[i], index_col='date', parse_dates=True, usecols=date_features))
        date_raw = pd.concat(date_raw_list, axis=1)

    elif date_path is not None:
        date_raw = pd.read_csv(date_path, index_col='date', parse_dates=True, usecols=date_features)
    else:
        date_raw = None

    df = pd.read_csv(path)
    # -------------------------------------------------------------
    #   Extract 'date' attribute into year/month/day/hour for time features
    # -------------------------------------------------------------
    df['date'] = pd.to_datetime(df['date'])
    if date_raw is not None:
        extra_mark = date_feature(date_raw, df.copy())
    else:
        extra_mark = None

    # ---------------------------------------------
    #   Standardize data
    #   Preprocess features
    # ---------------------------------------------

    scaler = StandardScaler(with_mean=True, with_std=True)
    fields = df.columns.values
    data = scaler.fit_transform(df[fields[1:]].values)
    mean = scaler.mean_
    scale = scaler.scale_
    stamp = scaler.fit_transform(timefeature(df))
    if extra_mark is not None:
        stamp = np.hstack((stamp, extra_mark))  # Merge time features with extra data
    else:
        stamp = stamp

    args = kwargs.get('args')
    print('Time feature dimension:', stamp.shape[1])
    args.d_mark = stamp.shape[1]

    # ---------------------------------------------
    #   Split dataset into train, valid, and test
    #   data contains all features except time-related ones
    #   stamp contains only time-related features
    # ---------------------------------------------
    train_data = data[:int(0.6 * len(data)), :]
    valid_data = data[int(0.6 * len(data)):int(0.8 * len(data)), :]
    test_data = data[int(0.8 * len(data)):, :]

    train_stamp = stamp[:int(0.6 * len(stamp)), :]
    valid_stamp = stamp[int(0.6 * len(stamp)):int(0.8 * len(stamp)), :]
    test_stamp = stamp[int(0.8 * len(stamp)):, :]

    dim = train_data.shape[-1]

    return [train_data, train_stamp], [valid_data, valid_stamp], [test_data, test_stamp], mean, scale, dim
