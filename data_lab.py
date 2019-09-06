from label import *
from load_md import get_md_by_tick as get_md
from input_data import chafen
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv

# from IndexData import IndexData
path = '/data/nvme/yingda/3/'
path1 = '/data/nvme/yingda/2'
data_path = '/data/remoteDir/server_200/mem_data'
code_index_file = '/data/remoteDir/server_200/mem_data/.index/code.csv'


def estab_day_list(day_begin, day_end):
    date_list = []
    year_begin = day_begin // 10000
    month_begin = day_begin // 100 % 100
    date_begin = day_begin % 100
    year_end = day_end // 10000
    month_end = day_end // 100 % 100
    date_end = day_end % 100
    if year_end > year_begin:
        interval = (year_end - year_begin) * 372 + (month_end * 31 + date_end) - (month_begin * 31 + date_begin)
    else:
        interval = (month_end * 31 + date_end) - (month_begin * 31 + date_begin)
    # print(interval)
    if interval < 0:
        print("cannot get the right date interval")
    else:
        date_list.append(day_begin)
        date = date_begin
        month = month_begin
        year = year_begin
        if interval != 0:
            for i in range(interval):
                if date < 31:
                    date += 1
                elif month < 12:
                    month += 1
                    date = 1
                else:
                    year += 1
                    month = 1
                    date = 1
                day = year * 10000 + month * 100 + date
                date_list.append(day)
    return date_list


def estab_one_day_data_lab(tag, day_list, is_chafen=False, is_zero_mean=True, is_minus_open=True):
    tag_dtype_float64 = ['totoff', 'vol', 'totbid', 'amount']
    one_day_data = []

    if tag in tag_dtype_float64:
        dtype1 = 'float64'
    else:
        dtype1 = 'float32'
    for i, day in enumerate(day_list):
        '''
        check whether tick data exists 
        '''
        year = str(day // 10000).zfill(4)
        month = str(day // 100 % 100).zfill(2)
        date = str(day % 100).zfill(2)
        path_day = os.path.join(data_path, year, month, date)
        if not os.path.exists(path_day):
            one_day_data.append(np.zeros((1, 1)))
        else:
            df_temp = get_md(day, tag, dtype=dtype1).T
            array_data = df_temp.values  # codes * ticks
            if is_minus_open == True:
                array_open = get_md(day, 'open', dtype='float32').T.values
                array_data -= array_open
            fullfill_data(array_data)
            if is_zero_mean == True:
                array_data -= (np.mean(array_data, axis=1) * np.ones((array_data.shape[1], array_data.shape[0]))).T
            if is_chafen == True:
                array_data = chafen(tag, array_data)
            else:
                array_data = array_data[:, 1:]
            # print(array_data[np.nonzero(np.isnan(array_data[:,0])==False)[0], :])
            # min_temp = np.min(array_data[np.nonzero(np.isnan(array_data[:,0])==False)[0], :], axis = 0)
            # max_temp = np.max(array_data[np.nonzero(np.isnan(array_data[:,0])==False)[0], :], axis = 0)
            # print(min_temp.shape)
            one_day_data.append(array_data)
    return one_day_data


def estab_open_low_high(day_list):
    tag_olh = ['open', 'low', 'high']
    for i, day in enumerate(day_list):
        olh_array = np.zeros((3690, 3))
        '''
        check whether tick data exists 
        '''
        year = str(day // 10000).zfill(4)
        month = str(day // 100 % 100).zfill(2)
        date = str(day % 100).zfill(2)
        path_day = os.path.join(data_path, year, month, date)
        if not os.path.exists(path_day):
            continue
        for i2, tag in enumerate(tag_olh):
            df_temp = get_md(day, tag, dtype='float32').T
            array_data = df_temp.values
            olh_array[:, i2] = array_data[:, -1]
        # all_tag_data.append(all_tag_array)
        np.save(path + 'olh/' + str(day), olh_array)


def estab_all_tag_data(day_list):
    tag_all = ['totoff', 'vol', 'totbid', 'amount', 'last', 'trade', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3',
               'bid_vol4', 'bid_vol5', 'bid_vol6', 'bid_vol7', 'bid_vol8', \
               'bid_vol9', 'bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4', 'ask_vol5', 'ask_vol6',
               'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10']
    tag_accum = ['amount', 'trade', 'vol', 'totbid', 'totoff']
    tag_dtype_float64 = ['totoff', 'vol', 'totbid', 'amount']
    # all_tag_data = []
    for i, day in enumerate(day_list):
        all_tag_array = np.zeros((3690, len(tag_all) * 240))
        '''
        check whether tick data exists 
        '''
        year = str(day // 10000).zfill(4)
        month = str(day // 100 % 100).zfill(2)
        date = str(day % 100).zfill(2)
        path_day = os.path.join(data_path, year, month, date)
        if not os.path.exists(path_day):
            continue
        for i2, tag in enumerate(tag_all):
            if tag in tag_dtype_float64:
                dtype1 = 'float64'
            else:
                dtype1 = 'float32'
            df_temp = get_md(day, tag, dtype=dtype1).T
            array_data = df_temp.values  # codes * ticks 3627*4802
            # print(array_data[0,:])
            if re.search(r'ask[0-9]+', tag) != None or re.search(r'bid[0-9]+', tag) != None:
                array_open = get_md(day, 'open', dtype='float32').T.values
                array_data -= array_open
                array_data = array_data[:, 1:]
            elif tag in tag_accum:
                array_data = chafen(tag, array_data)
            else:
                array_data = array_data[:, 1:]
            fullfill_data(array_data)
            array_data = array_data[:, np.arange(0, 4800, 10)]
            # print(array_data.shape)
            all_tag_array[:, np.arange(240) * len(tag_all) + np.ones(240, dtype='int64') * i2] = array_data[:,
                                                                                                 np.arange(240)].copy()
            if i == 0:
                plt.figure(1, figsize=(15, 10))
                plt.plot(np.arange(array_data.shape[1]), array_data[0, :], c='blue')
                plt.xlabel('x', fontsize=30)
                plt.ylabel('y', fontsize=30)
                plt.title('view of ' + tag, fontsize=40)
                plt.savefig(tag)
                plt.close(1)
                # all_tag_data.append(all_tag_array)
        np.save(path + str(day), all_tag_array)
    # plt.figure(1, figsize=(15, 10))
    # plt.plot(np.arange(all_tag_data[0].shape[1]), all_tag_data[0][0,:],c='blue')
    # plt.xlabel('x', fontsize = 30)
    # plt.ylabel('y', fontsize = 30)
    # plt.title('view', fontsize = 40)
    # plt.savefig('view')
    # plt.close(1)
    # return all_tag_data


def estab_x_data_lab(one_day_data):
    one_day_data2 = []
    for data in one_day_data:
        if data.shape[0] == 1:
            continue
        else:
            one_day_data2.append(data)
    return one_day_data2


def estab_text_like_data_lab(one_day_data, window_size):
    one_day_data = []
    for day in day_list:
        path_data = path + str(day) + '.npy'
        if not os.path.exists(path_data):
            continue
        array_data = np.load(path_data)
        one_day_data.append(array_data)
    num_days = len(one_day_data)
    num_codes = one_day_data[0].shape[0]
    text_like_data = []
    # sq_data = []
    for k in range(num_codes):
        for i in range(num_days - window_size + 1):
            one_text_data = []
            for j in range(window_size):
                if np.isnan(one_day_data[i + j][k, 0]):
                    # print(one_day_data[i+j][k])
                    continue
                    one_text_data.append(one_day_data[i + j][k])
            text_like_data.append(one_text_data)
    return text_like_data


def estab_pred_training_data(day_list, label_data, weight, is_transform=False):
    all_tag_data = []
    for day in day_list:
        path_data = path + str(day) + '.npy'
        if not os.path.exists(path_data):
            continue
        array_data = np.load(path_data)
        all_tag_data.append(array_data)
    codes_index1 = label_data[:, 0]  # array
    label_data = label_data[:, 1:]
    num_days = len(all_tag_data)
    codes_index2 = pd.read_csv(code_index_file, dtype={'stock_code': str}).set_index('stock_code')['idx']  # series
    num_codes = all_tag_data[0].shape[0]
    x_t = None
    y_t = None
    for j, code in enumerate(codes_index1):
        if not code in codes_index2.keys():
            continue
        x1 = []
        x = []
        y = []
        for i in range(num_days):
            if is_transform:
                x1.append(all_tag_data[i][j].dot(weight))
            else:
                x1.append(all_tag_data[i][j])  # have some problem
            y.append(label_data[j][i])
            if i >= 4 and i < num_days - 1:
                x.append(x1[i - 4:i + 1])
        x = np.array(x)
        y = np.array(y)
        y = y[5:]
        if j == 0:
            x_t = x
            y_t = y
        else:
            x_t = np.vstack((x_t, x))
            y_t = np.hstack((y_t, y))

    # print(x_t, y_t)
    print(x_t.shape)
    print(y_t.shape)
    np.save(path + 'x', x_t)
    np.save(path + 'y', y_t)

    return x_t, y_t


def fullfill_data(array_data):
    num_codes, num_ticks = array_data.shape
    for i in range(num_codes):
        num_nan = len(np.nonzero(np.isnan(array_data[i, :]) == True)[0])
        if num_nan < num_ticks / 10:
            meanVal = np.mean(array_data[i, np.nonzero(np.isnan(array_data[i, :]) == False)[0]])
            array_data[i, np.nonzero(np.isnan(array_data[i, :]))[0]] = meanVal
        else:
            array_data[i, :] = np.nan
    return 0


if __name__ == '__main__':
    tag = 'ask1'
    # day_list = [20190401, 20190402,20190403,20190404,20190405, 20190406, 20190407, 20190408, 20190409, 20190410, 20190411, 20190412]
    day_list = estab_day_list(20190601, 20190630)
    # one_day_data = estab_one_day_data_lab(tag, day_list)
    # text_like_data = estab_text_like_data_lab(one_day_data, 3)
    # one_day_data = estab_x_data_lab(one_day_data)
    estab_all_tag_data(day_list)
    # estab_open_low_high(day_list)
    # label_data = get_lab(path=path1, start='20190401',end='20190430').values
    # estab_pred_training_data(day_list, label_data, 0, is_transform = False)


