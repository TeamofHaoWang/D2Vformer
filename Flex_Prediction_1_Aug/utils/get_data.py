import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import numpy as np


def timefeature(dates):
    #-------------------------------------------------------------
    #   通过‘date’属性来对于时间顺序进行一个维度扩增
    #   目的是为了后续的embedding操作
    #   输入数据集必须有'date'属性，要求是年/月/日/时，如果某一个时刻没有，则修改下面的对应特征即可
    #-------------------------------------------------------------


    #---------------------------------------------
    #   经过 pd.to_datetime 自动处理年/月/日后
    #   通过行操作 .hour，.day等可以直接提取对应时间信息
    #---------------------------------------------


    dates["hour"] = dates["date"].apply(lambda row: row.hour / 23 - 0.5, 1)  # 一天中的第几小时
    dates["weekday"] = dates["date"].apply(lambda row: row.weekday() / 6 - 0.5, 1)  # 周几
    dates["day"] = dates["date"].apply(lambda row: row.day / 30 - 0.5, 1)  # 一个月的第几天
    dates["month"] = dates["date"].apply(lambda row: row.month / 365 - 0.5, 1)  # 一年的第几天
    return dates[["hour", "weekday", "day", "month"]].values


def date_feature(date_raw,df):
    date_day = df['date'].dt.floor('D')
    # for i in range(0,len(df['date'])):
    #     date_day = pd.to_datetime(str(df['date'][i].year)+'-'+str(df['date'][i].month)+'-'+str(df['date'][i].day))
    df['date_day'] = date_day

    extra_mark = []

    for i in range(0,len(df['date'])):
        date_feature = date_raw.loc[df['date_day'][i]]  # 取出一个Series
        data_mark = np.array(date_feature)
        extra_mark.append(data_mark)

    extra_mark = np.array(extra_mark)


    return extra_mark



def get_data(path,date_path,**kwargs):
    args = kwargs.get('args')
    date_features = ['date', 'abs_days', 'year', 'day', 'year_day', 'week', 'lunar_year',
                     'lunar_month', 'lunar_day', 'lunar_year_day', 'dayofyear', 'dayofmonth',
                     'monthofyear', 'dayofweek', 'dayoflunaryear', 'dayoflunarmonth',
                     'monthoflunaryear', 'holidays', 'workdays', 'residual_holiday',
                     'residual_workday', 'jieqiofyear', 'jieqi_day', 'dayofjieqi']
    if isinstance(date_path,list):
        date_raw_list=[]
        for i in range(len(date_path)):
            date_raw_list.append(pd.read_csv(date_path[i], index_col='date', parse_dates=True,
                                   usecols=date_features))
        date_raw = pd.concat(date_raw_list,axis=1)

    elif date_path!=None:
        date_raw = pd.read_csv(date_path, index_col='date', parse_dates=True,
                              usecols=date_features)
    else:
        date_raw=None

    df = pd.read_csv(path)

    if args.data_name == 'ETTm1':
        # 假设 df 已经存在
        n = len(df) // 2  # 一半的行数
        # 前一半
        df = df.iloc[:n]

    #-------------------------------------------------------------
    #   提取’date‘属性中的年/月/日/时
    #-------------------------------------------------------------
    df['date'] = pd.to_datetime(df['date'])
    if not date_raw is None:
        extra_mark = date_feature(date_raw,df.copy())
    else:
        extra_mark=None

    #---------------------------------------------
    #   标准化
    #   对各个特征数据进行预处理
    #---------------------------------------------

    scaler = StandardScaler(with_mean=True,with_std=True)
    # ---------------------------------------------
    #   特征的命名需要满足如下的条件：
    #   对于不同的数据集可以在这里进行修改。
    #   这里以后需要改进成通用的格式
    #   通过直接获取列名称的方式，改的更为通用。
    # ---------------------------------------------

    fields = df.columns.values
    # data = scaler.fit_transform(df[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']].values)
    data = scaler.fit_transform(df[fields[1:]].values)
    mean = scaler.mean_
    scale = scaler.scale_
    stamp = scaler.fit_transform(timefeature(df))
    if not extra_mark is None:
        stamp = np.hstack((stamp,extra_mark))  # 维度23 + 4
    else:
        stamp =stamp

    print('时间特征的维度：', stamp.shape[1])
    args.d_mark = stamp.shape[1]

    #---------------------------------------------
    #   划分数据集
    #   data是包含除时间外的特征
    #   stamp只包含时间特征
    #---------------------------------------------
    train_data = data[:int(0.6 * len(data)), :]
    valid_data = data[int(0.6 * len(data)):int(0.8 * len(data)), :]
    test_data = data[int(0.8 * len(data)):, :]

    train_stamp = stamp[:int(0.6 * len(stamp)), :]
    valid_stamp = stamp[int(0.6 * len(stamp)):int(0.8 * len(stamp)), :]
    test_stamp = stamp[int(0.8 * len(stamp)):, :]

    dim = train_data.shape[-1]

    return [train_data, train_stamp], [valid_data, valid_stamp], [test_data, test_stamp],mean,scale,dim

