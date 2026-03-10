# coding: utf-8
import csv

'''
写读追加状态
'r'：读
'w'：写
'a'：追加
'r+' == r+w（可读可写，文件若不存在就报错(IOError)）
'w+' == w+r（可读可写，文件若不存在就创建）
'a+' ==a+r（可追加可写，文件若不存在就创建）
对应的，如果是二进制文件，就都加一个b就好啦：
'rb'　　'wb'　　'ab'　　'rb+'　　'wb+'　　'ab+'
'''


# read csv file, return list
def read_csv(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


# write csv file, return nothing
def write_csv(file_name, data, mode='w+'):
    with open(file_name, mode, newline="\n", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)


# dict writ into csv file, return nothing
def write_csv_dict(file_name, data, mode='w+'):
    with open(file_name, mode, newline="\n", encoding='utf-8') as f:
        writer = csv.DictWriter(f, data[0].keys())
        # writer.writeheader()
        writer.writerows(data)


# dict read from csv file, return list
def read_csv_dict(file_name, mode='r'):
    with open(file_name, mode, newline="\n", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data