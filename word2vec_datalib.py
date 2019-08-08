# _*_ coding: utf-8 _*_
import numpy as np
from load_md import get_md_by_tick as get_md
# from input_data import chafen
import re
import os.path
import pandas as pd
from estab_day_list1 import estab_day_list 

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

# def read_day_tick_data(date):
#     path = '/data/dataDisk1/mulan/work/work1/'
#     tag_all = ['last', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5', 'bid_vol6',
#                'bid_vol7', 'bid_vol8', 'bid_vol9','bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4',
#                'ask_vol5', 'ask_vol6', 'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10', 'totoff', 'totbid',
#                'amount', 'vol', 'trade']
#     #tag_all = ['last', 'bid1']
#     tag_accum = ['amount', 'trade', 'vol', 'totbid', 'totoff']
#     tag_dtype_int64 = ['totoff', 'vol', 'totbid', 'amount']
#     num_tag = len(tag_all)
#     num_codes = 3627
#     num_ticks = 481
#     one_day_array_data = np.zeros((num_tag,3627,num_ticks))
#     stock_code_path = '/data/remoteDir/server_200/mem_data'
#     col_path = os.path.join(stock_code_path, '.index/code.csv')
#     col = pd.read_csv(col_path, dtype={'stock_code': str}).set_index('stock_code')['idx']
#     codes_index = col.index.tolist()       #record the location of codes having Nan
#     print(len(codes_index))
#     #codes = read_day_universe_stock(date)
#     #print(codes)
#     ticks = read_every_ten_ticks()
#     #print(ticks)
#     #array_open = get_md(date, 'open', ticks=ticks, codes=codes, dtype='float32').T
#     #array_open = get_md(date, 'open', ticks=ticks, codes=None, dtype='float32').T
#     for i,tag in enumerate(tag_all):
#         if tag in tag_dtype_int64:
#             dtype1 = 'int64'
#         else:
#             dtype1 = 'float32'
#         array_data1 = get_md(date, tag, ticks=ticks, codes=None, dtype=dtype1).T.values #code*ticks
#         #array_data = array_data.sample(n=481,axis=0)
#         #array_data1 = array_data.T.values 
#         # print(array_data[0,:])
#         # if re.search(r'ask[0-9]+', tag) != None or re.search(r'bid[0-9]+', tag) != None:
#         #     array_open = get_md(date, 'open', ticks=ticks, codes=codes, dtype='float32').T.values
#         #     array_data1 -= array_open[:,:]
#         #     array_data1 = array_data1[:, 1:]
#         if tag in tag_accum:
#             array_data1 = chafen(tag, array_data1)
#         else:
#             array_data1 = array_data1[:, 1:]
#         array_data1 = fullfill_data(array_data1)
#         one_day_array_data[i] = array_data1
#         #print(one_day_array_data)
#     windowsize = 11
#     #for x
#     codes_record = []
#     l = 0
#     # num_codes = 100
#     x2 = []
#     for j in range(num_codes):
#         if len(np.nonzero(np.isnan(one_day_array_data[:,j,:]) == True)[0]) > 0:
#         # if numpy.isnan(one_day_array_data[:,j,:]).any() == True:
#         # if one_day_array_data[:,j,:].hasNaN() == True:
#             l += 1
#             print(l)
#         else:
#             codes_record.append(codes_index[j])
#             for i in range(num_ticks-windowsize+1):
#                 one_text_array = []
#                 for k in range(windowsize):
#                     if k != 5:
#                         one_text_array.append(one_day_array_data[:,j,i+k])
#                 #print(one_text_array)
#                 x2.append(one_text_array)    
#     data_x = []
#     for i in range(len(codes_record)*(num_ticks-windowsize+1)):
#         x1 = []
#         for j in range(windowsize-1):
#             for k in range(num_tag):
#                 x1.append(x2[i][j][k])
#         data_x.append(x1)
#     # train_x = np.array(x2)
#     #for y
#     data_y = []
#     l = 0
#     for j in range(num_codes):
#         if codes_index[j] not in codes_record:
#             l += 1
#             print(l)
#         else:
#             for i in range(5,(num_ticks-windowsize+1)+5):
#                 one_output_array = one_day_array_data[:,j,i]
#                 data_y.append(one_output_array)
#     # train_y = np.array(data_y)
#     # np.save(path + str(date) + 'w2v_x', train_x)
#     # np.save(path + str(date) + 'w2v_y', train_y)
#     # print(codes_record)
#     # print(type(codes_record[0]))
#     x = np.array(data_x)
#     y = np.array(data_y)
#     print(x.shape)
#     print(y.shape)
#     return codes_record, x, y

# def fullfill_data(array_data):
#     num_codes, num_ticks = array_data.shape
#     for i in range(num_codes):
#         num_nan = len(np.nonzero(np.isnan(array_data[i, :]) == True)[0])
#         if num_nan < num_ticks / 10:
#             meanVal = np.mean(array_data[i, np.nonzero(np.isnan(array_data[i, :]) == False)[0]])
#             array_data[i, np.nonzero(np.isnan(array_data[i, :]))[0]] = meanVal
#         else:
#             array_data[i, :] = np.nan
#     return array_data

def chafen(tag, array_data):
    tag_accum = ['amount', 'trade', 'vol', 'totbid', 'totoff']
    # if re.search(r'ask[0-9]+', tag) != None or re.search(r'bid[0-9]+', tag) != None:
    #     data1 = array_data[:, :-1]
    #     data2 = array_data[:, 1:]
    #     return (data2 - data1) / data1
    if tag in tag_accum:
        data1 = array_data[:-1, :]
        data2 = array_data[1:, :]
        return data2 - data1
    else:
        return array_data[1:, :]

def fullfill_data(array_data):
    num_ticks, l = array_data.shape
    num_nan = len(np.nonzero(np.isnan(array_data[:, 0]) == True))
    if num_nan < num_ticks / 10:
        meanVal = np.mean(array_data[np.nonzero(np.isnan(array_data[:,0]) == False),0])
        array_data[np.nonzero(np.isnan(array_data[:, 0])),0] = meanVal
    else:
        array_data[:, 0] = np.nan
    return array_data

def read_codes_index():
    stock_code_path = '/data/remoteDir/server_200/mem_data'
    col_path = os.path.join(stock_code_path, '.index/code.csv')
    col = pd.read_csv(col_path, dtype={'stock_code': str}).set_index('stock_code')['idx']
    codes_index = col.index.tolist()
    return codes_index  

def save_day_tick_data(date,codes_index):
    path = '/data/dataDisk1/mulan/pre_data_201811/'
    tag_all = ['last', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5', 'bid_vol6',
               'bid_vol7', 'bid_vol8', 'bid_vol9','bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4',
               'ask_vol5', 'ask_vol6', 'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10', 'totoff', 'totbid',
               'amount', 'vol', 'trade']
    # tag_all = ['last', 'vol']
    tag_accum = ['amount', 'trade', 'vol', 'totbid', 'totoff']
    tag_dtype_int64 = ['totoff', 'vol', 'totbid', 'amount']
    num_tag = len(tag_all)
    num_codes = 3627 
    ticks = read_every_ten_ticks() #482
    p = 0
    # codes_index = codes_index[:5]
    # codes_record = []
    for i,code in enumerate(codes_index):
        # print(code)
        p += 1
        print(p)
        code_list = []
        code_list.append(code)
        # l = 0
        # m = 0
        for i,tag in enumerate(tag_all):
            if tag in tag_dtype_int64:
                dtype1 = 'int64'
            else:
                dtype1 = 'float32'
            if i==0:
                tag_array1 = get_md(date, tag, ticks=ticks, codes=code_list, dtype=dtype1).values #1*tick
                tag_array1 = chafen(tag, tag_array1)
                tag_array2 = fullfill_data(tag_array1)
                # if len(np.nonzero(np.isnan(tag_array2[:,0]) == True)[0]) > 0:
                #     # l += 1
                #     # print(code+ '_' + str(l) + "Nan")
                #     codes_index.pop(i)
                #     break
                # else:
                #     # m += 1
                #     # print(code+ '_' + str(m) + "Full")
                tag_array = tag_array2[:,:]
            else: 
                tag_array_add1 = get_md(date, tag, ticks=ticks, codes=code_list, dtype=dtype1).values #1*tick
                tag_array_add1 = chafen(tag, tag_array_add1)
                tag_array_add2 = fullfill_data(tag_array_add1)
                # print(tag_array.shape)
                # print(tag_array_add2.shape)
                # if len(np.nonzero(np.isnan(tag_array_add2[:,0]) == True)[0]) > 0:
                #     # l += 1
                #     # print(code+ '_' + str(l) + "Nan")
                #     codes_index.pop(i)
                #     break
                # else:
                #     # m += 1
                #     # print(code+ '_' + str(m) + "Full")
                tag_array = np.hstack((tag_array,tag_array_add2))
        # if code in codes_index:
            # print(p)
            # print(l)
            # print(tag_array.shape)
            # print(tag_array)
            np.save(path + code + '_' + str(date), tag_array)
    # print(len(codes_index))
    # return codes_record

def read_day_tick_data(date, codes_index):
    path = '/data/dataDisk1/mulan/pre_data4/'
    tag_all = ['last', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5', 'bid_vol6',
               'bid_vol7', 'bid_vol8', 'bid_vol9','bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4',
               'ask_vol5', 'ask_vol6', 'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10', 'totoff', 'totbid',
               'amount', 'vol', 'trade']
    # tag_all = ['last', 'vol']
    num_tag = len(tag_all)
    num_codes = len(codes_index)
    num_ticks = 482
    windowsize = 11
    codes_record = []
    l = 0
    list_x = np.zeros((num_codes*(num_ticks-windowsize),num_tag*(windowsize-1)))
    list_y = np.zeros((num_codes*(num_ticks-windowsize),num_tag))
    for i,code in enumerate(codes_index):
        read_path = os.path.join(path, code + '_' + str(date) + '.npy')
        # data_array = np.memmap(read_path,dtype='float64',mode='r',shape=(num_ticks-1,num_tag))
        data_array = np.load(read_path)
        if len(np.nonzero(np.isnan(data_array[:,:]) == True)[0]) > 0:
            l += 1
            # print(l)
        else:
            codes_record.append(codes_index[i])
            #for x
            for j in range(num_ticks-windowsize):
                one_text_x = np.zeros((windowsize-1)*num_tag)
                # print(one_text_x)
                for k in range(windowsize):
                    if k < 5:
                        one_text_x[k*num_tag:(k+1)*num_tag] = data_array[j+k,:]
                    elif k == 5:
                        g = 0
                    else:
                        one_text_x[(k-1)*num_tag:k*num_tag] = data_array[j+k,:]
                list_x[(i-l)*(num_ticks-windowsize)+j,:] = one_text_x[:]        
            # for y 
            for h in range(5,(num_ticks-windowsize)+5):
                # list_y = list_y.append(data_array.iloc[h,1:num_tag+1],ignore_index=True)
                # print(i*(num_ticks-windowsize)+h)
                list_y[(i-l)*(num_ticks-windowsize)+h-5,:] = data_array[h,:]
                # list_y.loc[0,0] = data_array.iloc[h,1]
                # list_y.loc[0,1] = data_array.iloc[h,2]
        # data_array._mmap.close()
    print(list_x.shape)
    print(list_y.shape)
    list_x = list_x[:(num_codes-l)*(num_ticks-windowsize),:]
    list_y = list_y[:(num_codes-l)*(num_ticks-windowsize),:]
    print(list_x.shape)
    print(list_y.shape)
    return codes_record, list_x, list_y
    

if __name__ == "__main__":
    codes_index = read_codes_index()
    date_list = estab_day_list(20181126, 20181130)
    data_path = '/data/remoteDir/server_200/mem_data/'
    for date in date_list:
        print(date)
        year = str(date // 10000).zfill(4)
        month = str(date // 100 % 100).zfill(2)
        day= str(date% 100).zfill(2)    
        path_day = os.path.join(data_path,year,month,day,'open')
        if not os.path.exists(path_day):
            flag = 0
            print(str(date) + 'not exit')
        else:
            save_day_tick_data(date, codes_index)
