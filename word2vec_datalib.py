# _*_ coding: utf-8 _*_
import numpy as np
from load_md import get_md_by_tick as get_md
from input_data import chafen
import re
import os.path
import pandas as pd


def read_day_universe_stock(date):
    path1 = '/data/remoteDir/server_200/universe/'
    codes = []
    stock_path = os.path.join(path1, str(date))
    with open(stock_path,'r') as f:
        lines = f.readlines()
        for i in range(2000):
            line = lines[i].split()
            codes.append(line[0])
    return codes


def read_every_ten_ticks():
    data_path ='/data/remoteDir/server_200/mem_data' 
    inx_path = os.path.join(data_path ,'.index/tick.csv')
    inx = pd.read_csv(inx_path).set_index('tick_time')['idx']
    ticks = inx.index.tolist()
    re_ticks = []
    for time_interval in range(1,3):
        re_ticks.append(ticks[(time_interval-1)*2401])
        for i in range(240):
            re_ticks.append(ticks[time_interval*i*10+1])
    return re_ticks

def read_day_tick_data(date):
    path = '/data/dataDisk1/mulan/work/'
    tag_all = ['last', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5', 'bid_vol6',
               'bid_vol7', 'bid_vol8', 'bid_vol9','bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4',
               'ask_vol5', 'ask_vol6', 'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10', 'totoff', 'totbid',
               'amount', 'vol', 'trade', 'low', 'high', 'open']
    #tag_all = ['last', 'bid1']
    tag_accum = ['amount', 'trade', 'vol', 'totbid', 'totoff']
    tag_dtype_int64 = ['totoff', 'vol', 'totbid', 'amount']
    num_tag = len(tag_all)
    num_codes = 2000
    #num_codes = 1
    num_ticks = 481
    one_day_array_data = np.zeros((num_tag,2000,num_ticks))
    data_x = []
    codes = read_day_universe_stock(date)
    ticks = read_every_ten_ticks()
    array_open = get_md(date, 'open', ticks=ticks, codes=codes, dtype='float32').T
    for i,tag in enumerate(tag_all):
        if tag in tag_dtype_int64:
            dtype1 = 'int64'
        else:
            dtype1 = 'float32'
        array_data1 = get_md(date, tag, ticks=ticks, codes=codes, dtype=dtype1).T.values #code*ticks
        #array_data = array_data.sample(n=481,axis=0)
        #array_data1 = array_data.T.values 
        # print(array_data[0,:])
        # if re.search(r'ask[0-9]+', tag) != None or re.search(r'bid[0-9]+', tag) != None:
        #     array_open = get_md(date, 'open', ticks=ticks, codes=codes, dtype='float32').T.values
        #     array_data1 -= array_open[:,:]
        #     array_data1 = array_data1[:, 1:]
        if tag in tag_accum:
            array_data1 = chafen(tag, array_data1)
        else:
            array_data1 = array_data1[:, 1:]
        array_data1 = fullfill_data(array_data1)
        one_day_array_data[i] = array_data1
        #print(one_day_array_data)
    windowsize = 11
    #for x
    record = [] #record the location of codes having Nan
    l = 0
    for j in range(num_codes):
        if len(np.nonzero(np.isnan(one_day_array_data[:,j,:]) == True)[0]) > 0:
        # if numpy.isnan(one_day_array_data[:,j,:]).any() == True:
        # if one_day_array_data[:,j,:].hasNaN() == True:
            record.append(j)
            l += 1
            print(l)
        else:
            for i in range(num_ticks-windowsize+1):
                one_text_array = []
                for k in range(windowsize):
                    if k != 5:
                        one_text_array.append(one_day_array_data[:,j,i+k])
                #print(one_text_array)
                data_x.append(one_text_array)    
    x2 = []
    for i in range((num_codes-len(record))*(num_ticks-windowsize+1)):
        x1 = []
        for j in range(windowsize-1):
            for k in range(num_tag):
                x1.append(data_x[i][j][k])
        x2.append(x1)
    train_x = np.array(x2)
    #for y
    data_y = []
    l = 0
    for j in range(num_codes):
        if j in record:
            l += 1
            print(l)
        else:
            for i in range(5,(num_ticks-windowsize+1)+5):
                one_output_array = one_day_array_data[:,j,i]
                data_y.append(one_output_array)
    train_y = np.array(data_y)
    print(train_x.shape)
    print(train_y.shape)
    np.save(path + str(date) + 'w2v_x', train_x)
    np.save(path + str(date) + 'w2v_y', train_y)

def fullfill_data(array_data):
    num_codes, num_ticks = array_data.shape
    for i in range(num_codes):
        num_nan = len(np.nonzero(np.isnan(array_data[i, :]) == True)[0])
        if num_nan < num_ticks / 10:
            meanVal = np.mean(array_data[i, np.nonzero(np.isnan(array_data[i, :]) == False)[0]])
            array_data[i, np.nonzero(np.isnan(array_data[i, :]))[0]] = meanVal
        else:
            array_data[i, :] = np.nan
    return array_data

if __name__ == "__main__":
    read_day_tick_data(20171030)
