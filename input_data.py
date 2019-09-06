from label import get_lab
from load_md import get_md_by_tick as get_md
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import re
from scipy.fftpack import fft,ifft
path = '/home/mulan/mulan/preprocessing1'
path2 = '/home/mulan/mulan/preprocessing1/fft_fig/'


def normalization(array_data):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(array_data)


def chafen(tag, array_data):
    tag_accum = ['amount', 'trade', 'vol', 'totbid', 'totoff']
    # if re.search(r'ask[0-9]+', tag) != None or re.search(r'bid[0-9]+', tag) != None:
    #     data1 = array_data[:, :-1]
    #     data2 = array_data[:, 1:]
    #     return (data2 - data1) / data1
    if tag in tag_accum:
        data1 = array_data[:, :-1]
        data2 = array_data[:, 1:]
        return data2 - data1
    else:
        return array_data


def read_data_by_tag():
    '''
    tag_all = ['totoff', 'vol', 'totbid', 'amount', 'last', 'low', 'high', 'open', 'avebid', 'aveoff', 'trade', 'bid1',
               'bid2',
               'bid3', 'bid4', 'bid5', 'bid6', 'bid7', 'bid8', 'bid9', 'bid10', 'ask1', 'ask2', 'ask3', 'ask4', 'ask5',
               'ask6', 'ask7', 'ask8', 'ask9', 'ask10', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5',
               'bid_vol6', 'bid_vol7', 'bid_vol8', 'bid_vol9',
               'bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4', 'ask_vol5', 'ask_vol6', 'ask_vol7',
               'ask_vol8', 'ask_vol9',
               'ask_vol10']
    '''
    tag_dtype_int64 = ['totoff', 'vol', 'totbid', 'amount']

    # print(len(tag_all))
    day_list = [20160504, 20160505, 20160506]
    # start_day = 20160101
    # end_day = 20160131
    # day_list = IndexData().get_deal_day_list_in_period(start_day,end_day)
    # print(day_list)
    data_norm = {}
    data_cha = {}
    tag_all = ['ask1']    
    for tag in tag_all:
        #file_path = path + '/' + tag + '.txt'
        # file_path2 = path + '/' + tag + '_norm' + '.txt'
        if tag in tag_dtype_int64:
            dtype1 = 'int64'
        else:
            dtype1 = 'float32'
        i = 0
        for date in day_list:
            df_temp = get_md(date, tag, dtype=dtype1).T
            array_data_temp = df_temp.values
            # print(array_data_temp)
            if i == 0:
                array_data = array_data_temp.copy()
            else:
                array_data = np.hstack((array_data, array_data_temp))
                # print(array_data.shape)
            # array_data (3627, 4802, 3)
            i += 1
        nor_data = normalization(array_data)
        data_norm[tag] = nor_data
        cha_data = chafen(tag, array_data)
        data_cha[tag] = cha_data
        #np.savetxt(file_path, cha_data, delimiter=',')
        # np.savetxt(file_path2, nor_data,delimiter=',')
        plt.figure(1, figsize=(15, 10))
        plt.plot(np.arange(4801), cha_data[0, 0:4801], c='blue')
        # if tag=='vol':
        #         print(nor_data[0,0:4801])
        plt.xlabel('x', fontsize=30)
        plt.ylabel('y', fontsize=30)
        plt.title(tag, fontsize=40)
        plt.savefig('chafen_fig/' + tag + '_' + 'chafen')
        plt.close(1)

    return data_norm, data_cha

'''
def read_chafen_data(tag):
    file_path = path + '/' + tag + '.txt'
    file = open(file_path,'r')
    chafen_data = np.loadtxt(file,delimiter=',')
    return chafen_data
'''

def fft_for_chafen_data(tag, chafen_data):
    num_codes, num_ticks = chafen_data.shape
    x=np.arange(2400)
    for i in range(int((num_ticks+1)/2401)):
        data_in_two_hours = chafen_data[0,i*2401:(i+1)*2401]-sum(chafen_data[0,i*2401:(i+1)*2401])/len(chafen_data[0,i*2401:(i+1)*2401])
        # print(data_in_two_hours)
        fft_data_in_two_hours = fft(data_in_two_hours)
        # print(fft_data_in_two_hours)
        real = fft_data_in_two_hours.real
        imag = fft_data_in_two_hours.imag
        abs_fft_data_in_two_hours=abs(fft_data_in_two_hours)
        fft_y_norm = abs_fft_data_in_two_hours/2400
        plt.figure(1)
        plt.plot(x,fft_y_norm[0:2400],'r')
        plt.title('fft'+tag,fontsize=9,color='b')
        plt.savefig(path2+tag+'_'+'chafen'+str(i))
        plt.close(1)
    return 0

if __name__ == '__main__':
    tag='ask1'
    chafen_data = read_data_by_tag()
    # print(chafen_data[0,:4801])
    fft_for_chafen_data(tag, chafen_data)

    