import os
import numpy as np
import pandas as pd
from estab_day_list import estab_day_list 

def trans_to_low_dim(x, W, alpha=0.3):
    x_t = x.dot(W)
    x_t[np.nonzero(x_t[:] < 0)[0]] /= 0.3
    return x_t

# def fill_zero(date_list):
#     stock_code_path = '/data/remoteDir/server_200/mem_data'
#     col_path = os.path.join(stock_code_path, '.index/code.csv')
#     col = pd.read_csv(col_path, dtype={'stock_code': str}).set_index('stock_code')['idx']
#     codes_index = col.index.tolist()       
#     num_codes = len(codes_index)
#     num_ticks1 = 471
#     num_tag_reduce = 14
#     data_array1 = np.zeros((num_codes*num_ticks1,num_tag_reduce))
#     for date in date_list:
#         path1 = '/data/dataDisk1/mulan/final_data1/'
#         path2 = '/data/dataDisk1/mulan/final_data2/'
#         path3 = '/data/dataDisk1/mulan/final_data3/'
#         try:
#             data_array = np.load(path1 + str(date) +'.npy')
#         except OSError as reason:
#             print('file not exists' + str(reason))
#         else:
#             # print(data_array.shape)
#             code_record = np.load(path2 + str(date) +'_codes.npy').tolist()  #record the location of codes having Nan
#             l = 0
#             for i,code in enumerate(codes_index):
#                 if code in code_record:
#                     data_array1[i*num_ticks1:(i+1)*num_ticks1,:] = data_array[(i-l)*num_ticks1:(i-l+1)*num_ticks1,:]
#                 else:
#                     l += 1
#                     data_array1[i*num_ticks1:(i+1)*num_ticks1,:] = np.zeros((num_ticks1,num_tag_reduce))
#             np.save(path3 + str(date) +'.npy', data_array1)

def fill_zero(date_list):
    path1 = '/data/dataDisk1/mulan/pre_data_201902/'
    path2 = '/data/dataDisk1/mulan/final_data2/'
    path3 = '/data/dataDisk1/mulan/final_data3/'
    tag_all = ['last', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5', 'bid_vol6',
               'bid_vol7', 'bid_vol8', 'bid_vol9','bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4',
               'ask_vol5', 'ask_vol6', 'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10', 'totoff', 'totbid',
               'amount', 'vol', 'trade']
    # tag_all = ['last', 'vol']
    stock_code_path = '/data/remoteDir/server_200/mem_data'
    col_path = os.path.join(stock_code_path, '.index/code.csv')
    col = pd.read_csv(col_path, dtype={'stock_code': str}).set_index('stock_code')['idx']
    codes_index = col.index.tolist()       
    num_codes = len(codes_index)
    num_tag = len(tag_all)
    num_tag1 = 14
    num_ticks = 482
    windowsize = 11
    list_x = np.zeros((num_codes*(num_ticks-windowsize),num_tag1))
    for date in date_list:
        weight = np.load(path2 + str(date) +'_weights.npy')
        for i,code in enumerate(codes_index):
            read_path = os.path.join(path1, code + '_' + str(date) + '.npy')
            data_array = np.load(read_path)
            if len(np.nonzero(np.isnan(data_array[:,:]) == True)[0]) > 0:
                for h in range(5,(num_ticks-windowsize)+5):
                    list_x[i*(num_ticks-windowsize)+h-5,:] = np.zeros(num_tag1)
                # print(l)
            else:   
                # for y 
                for h in range(5,(num_ticks-windowsize)+5):
                    list_x[i*(num_ticks-windowsize)+h-5,:] = trans_to_low_dim(data_array[h,:], weight)
        print(list_x.shape)
        np.save(path3 + str(date) +'.npy', list_x)            


if __name__ == "__main__":
    date_list = estab_day_list(20190225, 20190228)
    fill_zero(date_list)