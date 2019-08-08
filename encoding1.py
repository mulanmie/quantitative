import numpy as np
from load_md import get_md_by_tick as get_md
from input_data import chafen
import re
# import os.path
import pandas as pd
# from word2vec_datalib import read_day_tick_data
from estab_day_list1 import estab_day_list 

from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU
from keras.callbacks import EarlyStopping
from data_lab import *
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.01
# set_session(tf.Session(config=config))

def generate_training_data(text_like_data, max_text_size):
    text_num = len(text_like_data)
    train_x = []
    train_y = []
    for i in range(text_num):
        text_size = len(text_like_data[i])
        if text_size <= 3 * max_text_size / 4:
            continue
        if text_size < max_text_size \
                and text_size > 3 * max_text_size / 4:
            for k in range(max_text_size - text_size):
                text_like_data[i].append(text_like_data[i][text_size - 1])
        mid = int(max_text_size / 2)
        y = text_like_data[i][mid]
        x = text_like_data[i][0]
        for j in range(1, max_text_size):
            if j != mid:
                # x = np.vstack((x, text_like_data[i][j]))
                x = np.hstack((x, text_like_data[i][j]))
        # x /= max_text_size-1
        # print(y.shape)
        # print(x.shape)
        train_y.append(y)
        train_x.append(x)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    # train_x = (train_x-min_*np.ones((train_x.shape[0], train_x.shape[1])))/(max_-min_)
    # train_y = (train_y-min_*np.ones((train_x.shape[0], train_x.shape[1])))/(max_-min_)
    # np.savetxt('x.txt',train_x)
    # np.savetxt('y.txt', train_y)
    # print(train_x)
    # print(train_y)

    return train_x, train_y


def trans_to_low_dim(x, W, alpha=0.3):
    x_t = x.dot(W)
    x_t[np.nonzero(x_t[:] < 0)[0]] /= 0.3
    return x_t


class embedding_model:
    def __init__(self):
        self.model = Sequential()
        self.weight = None

    def establish(self, x_dim, y_dim):
        self.model.add(Dense(14, input_dim=x_dim, name='dense_1'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dense(units=y_dim, name='dense_2'))
        # self.model.add(LeakyReLU(alpha=0.05))
        self.model.compile(loss='mse',
                           optimizer='Adam',
                           metrics=['mse'])

    def train(self, train_x, train_y):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(train_x, train_y, epochs=100, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
        # self.model.save_weights('my_model_weights.h5')
        self.weight = self.model.get_weights()[2].T

    def predict(self, test_x):
        return self.model.predict(test_x)

def read_day_tick_data(date_list, codes_index):
    path = '/data/dataDisk1/mulan/pre_data_201902/'
    tag_all = ['last', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5', 'bid_vol6',
               'bid_vol7', 'bid_vol8', 'bid_vol9','bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4',
               'ask_vol5', 'ask_vol6', 'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10', 'totoff', 'totbid',
               'amount', 'vol', 'trade']
    # tag_all = ['last', 'vol']
    num_tag = len(tag_all)
    num_codes = len(codes_index)
    num_ticks = 482
    windowsize = 11
    codes_record = {}
    num_date = len(date_list)
    # list_x = np.zeros((num_codes*(num_ticks-windowsize),num_tag*(windowsize-1)))
    # list_y = np.zeros((num_codes*(num_ticks-windowsize),num_tag))
    for k, day in enumerate(date_list):
        codes_record[k] = []
        if k == 0:
            l = 0
            list_x1 = np.zeros((num_codes*(num_ticks-windowsize),num_tag*(windowsize-1)))
            list_y1 = np.zeros((num_codes*(num_ticks-windowsize),num_tag))
            for i,code in enumerate(codes_index):
                read_path = os.path.join(path, code + '_' + str(day) + '.npy')
                # data_array = np.memmap(read_path,dtype='float64',mode='r',shape=(num_ticks-1,num_tag))
                data_array = np.load(read_path)
                if len(np.nonzero(np.isnan(data_array[:,:]) == True)[0]) > 0:
                    l += 1
                    # print(l)
                else:
                    codes_record[k].append(codes_index[i])
                    #for x
                    for j in range(num_ticks-windowsize):
                        one_text_x = np.zeros((windowsize-1)*num_tag)
                        # print(one_text_x)
                        for m in range(windowsize):
                            if m < 5:
                                one_text_x[m*num_tag:(m+1)*num_tag] = data_array[j+m,:]
                            elif m == 5:
                                g = 0
                            else:
                                one_text_x[(m-1)*num_tag:m*num_tag] = data_array[j+m,:]
                        list_x1[(i-l)*(num_ticks-windowsize)+j,:] = one_text_x[:]        
                    # for y 
                    for h in range(5,(num_ticks-windowsize)+5):
                        # list_y = list_y.append(data_array.iloc[h,1:num_tag+1],ignore_index=True)
                        # print(i*(num_ticks-windowsize)+h)
                        list_y1[(i-l)*(num_ticks-windowsize)+h-5,:] = data_array[h,:]
                        # list_y.loc[0,0] = data_array.iloc[h,1]
                        # list_y.loc[0,1] = data_array.iloc[h,2]
                # data_array._mmap.close()
            print(list_x1.shape)
            print(list_y1.shape)
            list_x = list_x1[:(num_codes-l)*(num_ticks-windowsize),:]
            list_y = list_y1[:(num_codes-l)*(num_ticks-windowsize),:]
            print(list_x1.shape)
            print(list_y1.shape)
            list_x = list_x1[:,:]
            list_y = list_y1[:,:]
        else:
            l = 0
            list_x1 = np.zeros((num_codes*(num_ticks-windowsize),num_tag*(windowsize-1)))
            list_y1 = np.zeros((num_codes*(num_ticks-windowsize),num_tag))
            for i,code in enumerate(codes_index):
                read_path = os.path.join(path, code + '_' + str(day) + '.npy')
                # data_array = np.memmap(read_path,dtype='float64',mode='r',shape=(num_ticks-1,num_tag))
                data_array = np.load(read_path)
                if len(np.nonzero(np.isnan(data_array[:,:]) == True)[0]) > 0:
                    l += 1
                    # print(l)
                else:
                    codes_record[k].append(codes_index[i])
                    #for x
                    for j in range(num_ticks-windowsize):
                        one_text_x = np.zeros((windowsize-1)*num_tag)
                        # print(one_text_x)
                        for m in range(windowsize):
                            if m < 5:
                                one_text_x[m*num_tag:(m+1)*num_tag] = data_array[j+m,:]
                            elif m == 5:
                                g = 0
                            else:
                                one_text_x[(m-1)*num_tag:m*num_tag] = data_array[j+m,:]
                        list_x1[(i-l)*(num_ticks-windowsize)+j,:] = one_text_x[:]        
                    # for y 
                    for h in range(5,(num_ticks-windowsize)+5):
                        # list_y1 = list_y1.append(data_array.iloc[h,1:num_tag+1],ignore_index=True)
                        # print(i*(num_ticks-windowsize)+h)
                        list_y1[(i-l)*(num_ticks-windowsize)+h-5,:] = data_array[h,:]
                        # list_y1.loc[0,0] = data_array.iloc[h,1]
                        # list_y1.loc[0,1] = data_array.iloc[h,2]
                # data_array._mmap.close()
            print(list_x1.shape)
            print(list_y1.shape)
            list_x1 = list_x1[:(num_codes-l)*(num_ticks-windowsize),:]
            list_y1 = list_y1[:(num_codes-l)*(num_ticks-windowsize),:]
            print(list_x1.shape)
            print(list_y1.shape)
            list_x = np.vstack((list_x,list_x1[:,:]))
            list_y = np.vstack((list_y,list_y1[:,:])) 
    return codes_record, list_x, list_y

def fill_zero1(date):
    path1 = '/data/dataDisk1/mulan/pre_data_201902/'
    path2 = '/data/dataDisk1/mulan/final_data/'
    path3 = '/data/dataDisk1/mulan/final_data1/'
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
    weight = np.load(path2 + str(20190114) + '-' + str(20190118) + '.npy')
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


if __name__ == '__main__':
    '''
    #one_day_data = estab_one_day_data_lab(tag, day_list, is_chafen=False, is_zero_mean=False, is_minus_open=True)
    #text_like_data = estab_text_like_data_lab(one_day_data, 3)
    #x, y = generate_training_data(text_like_data, 3)
    # path1 = '/data/dataDisk1/mulan/work/work1/20171030w2v_x.npy'
    # path2 = '/data/dataDisk1/mulan/work/work1/20171030w2v_y.npy'
    # x = np.load(path1)
    # y = np.load(path2)
    date_list = estab_day_list(20190211, 20190215)
    data_path = '/data/remoteDir/server_200/mem_data/'
    path1 = '/data/dataDisk1/mulan/final_data1/'
    path2 = '/data/dataDisk1/mulan/final_data/'
    stock_code_path = '/data/remoteDir/server_200/mem_data'
    col_path = os.path.join(stock_code_path, '.index/code.csv')
    col = pd.read_csv(col_path, dtype={'stock_code': str}).set_index('stock_code')['idx']
    codes_index = col.index.tolist()       #record the location of codes having Nan
    # date_codes = []
    codes_record, x, y = read_day_tick_data(date_list,codes_index)
    num_codes = len(codes_index)
    windowsize = 10
    num_ticks = 471
    num_tag = 28
    num_tag_jiangwei =14
    embedding = embedding_model()
    embedding.establish(num_tag*windowsize, num_tag)
    embedding.train(x, y)
    # weight = np.zeros((4802, 32))
    # for i in range(32):
    #     weight[i,i]=1
    # print(len(weight))
    # model = embedding.model.load_weights('my_model_weights.h5')
    # np.savetxt('weight.txt',weight)
    # for i in range(10):
    #     plt.figure(1, figsize=(15, 10))
    #     plt.plot(np.arange(28), y[i*471][0:28], c='red')
    #     plt.title(str(i)+'_real', fontsize=40)
    #     plt.savefig(str(i)+'_real')
    #     plt.close(1)
    #     x_trans = trans_to_low_dim(y[i*470][0:28], embedding.weight)
    #     plt.figure(2, figsize=(15, 10))
    #     plt.plot(np.arange(14), x_trans, c='blue')
    #     plt.title(str(i)+'_real' + '_trans', fontsize=40)
    #     plt.savefig(str(i)+'_real' + '_trans')
    #     plt.close(2)
            # result_x = np.zeros((num_codes*num_ticks,num_tag_jiangwei))
            # l = 0
            # for i in range(num_codes):
            #     if codes_index[i] not in codes_record:
            #         l += 1
            #         print(l)
            #     else:
            #         for j in range(num_ticks):
            #             x_trans = trans_to_low_dim(y[(i-l)*num_ticks+j][0:num_tag], embedding.weight)
            #             result_x[(i-l)*num_ticks,:] = x_trans[:]
            # np.save(path1 + str(date),result_x[:len(codes_record)*num_ticks,:])
    # np.save(path2 + str(date_list[0]) + '-' + str(date_list[-1]), codes_record)
    print(embedding.weight)  
    np.save(path2 + str(date_list[0]) + '-' + str(date_list[-1]), embedding.weight) #weight(28,14)
    '''
    date_list = estab_day_list(20190201, 20190231)
    for date in date_list:
        print(date)
        fill_zero1(date)
    
