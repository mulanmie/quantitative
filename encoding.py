import numpy as np
from load_md import get_md_by_tick as get_md
from input_data import chafen
import re
# import os.path
import pandas as pd
# from word2vec_datalib import read_day_tick_data
from estab_day_list import estab_day_list 
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Activation, LeakyReLU, BatchNormalization
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
        # self.model.add(BatchNormalization(axis=1, input_shape =(x_dim,)))
        self.model.add(Dense(20, input_dim = (x_dim), name='dense_1'))
        self.model.add(LeakyReLU(alpha=0.3))
        self.model.add(Dense(units=y_dim, name='dense_3'))
        # self.model.add(LeakyReLU(alpha=0.05))
        self.model.compile(loss='mse',
                           optimizer='Adam',
                           metrics=['mse'])
        print(self.model.summary())

    def train(self, train_x, train_y):
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit(train_x, train_y, epochs=1, batch_size=64, validation_split=0.2, callbacks=[early_stopping])
        # print(self.model.get_weights()[0].shape)
        # print(self.model.get_weights()[1].shape)      
        # print(self.model.get_weights()[2].shape)
        # print(self.model.get_weights()[3].shape)
        # print(self.model.get_weights()[4].shape)
        # print(self.model.get_weights()[5].shape)
        # print(self.model.get_weights()[6].shape)
        # print(self.model.get_weights()[7].shape)
        # print(self.model.get_weights()[8].shape)

        self.weight = self.model.get_weights()[2].T

         # self.model.save_weights('my_model_weights.h5')

    def predict(self, test_x):
        return self.model.predict(test_x)

def read_day_tick_data(path,date, codes_index):
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    # path = '/data/dataDisk1/mulan/pre_data_201811/'
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
            # data_array = min_max_scaler.fit_transform(data_array)
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

def fill_zero(path1, path2, path3, date):
    # path1 = '/data/dataDisk1/mulan/pre_data_201903/'
    # path2 = '/data/dataDisk1/mulan/final_data6/'
    # path3 = '/data/dataDisk1/mulan/final_data7/'
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
    num_tag1 = 20
    num_ticks = 482
    windowsize = 11
    list_x = np.zeros((num_codes*(num_ticks-windowsize),num_tag1))
    weight = np.load(path2 + str(date) +'_weights.npy')
    print(date)
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
    
    #one_day_data = estab_one_day_data_lab(tag, day_list, is_chafen=False, is_zero_mean=False, is_minus_open=True)
    #text_like_data = estab_text_like_data_lab(one_day_data, 3)
    #x, y = generate_training_data(text_like_data, 3)
    # path1 = '/data/dataDisk1/mulan/work/work1/20171030w2v_x.npy'
    # path2 = '/data/dataDisk1/mulan/work/work1/20171030w2v_y.npy'
    # x = np.load(path1)
    # y = np.load(path2)
    # date_list = estab_day_list(20181106, 20181106)
    # data_path = '/data/remoteDir/server_200/mem_data/'
    # path1 = '/data/dataDisk1/mulan/pre_data_201811/'
    # path2 = '/data/dataDisk1/mulan/final_data2/'
    # path3 = '/data/dataDisk1/mulan/final_data3/'
    date_list = estab_day_list(20190201, 20190231)
    data_path = '/data/remoteDir/server_200/mem_data/'
    path1 = '/data/dataDisk1/mulan/pre_data_201902/'
    path2 = '/data/dataDisk1/mulan/final_data6/'
    path3 = '/data/dataDisk1/mulan/final_data7/'
    stock_code_path = '/data/remoteDir/server_200/mem_data'
    col_path = os.path.join(stock_code_path, '.index/code.csv')
    col = pd.read_csv(col_path, dtype={'stock_code': str}).set_index('stock_code')['idx']
    codes_index = col.index.tolist()       #record the location of codes having Nan
    # date_codes = []
    for date in date_list:
        year = str(date // 10000).zfill(4) 
        month = str(date // 100 % 100).zfill(2)
        day= str(date% 100).zfill(2)    
        path_day = os.path.join(data_path,year,month,day,'open')
        if not os.path.exists(path_day):
            flag = 0
            print(str(date) + 'not exit')
        else:
            codes_record, x, y = read_day_tick_data(path1,date,codes_index)
            num_codes = len(codes_index)
            windowsize = 10
            num_ticks = 471
            num_tag = 28
            num_tag_jiangwei =20
            embedding = embedding_model()
            embedding.establish(num_tag*windowsize, num_tag)
            embedding.train(x, y)
            np.save(path2 + str(date) + '_codes', codes_record)
            print(embedding.weight)  
            np.save(path2 + str(date) + '_weights', embedding.weight) #weight(28,14)
            fill_zero(path1, path2, path3, date)
    # weight = np.zeros((4802, 32))
    # for i in range(32):
    #     weight[i,i]=1
    # print(len(weight))
    # model = embedding.model.load_weights('my_model_weights.h5')
    # np.savetxt('weight.txt',weight)
    for i in range(10):
        plt.figure(1, figsize=(15, 10))
        plt.plot(np.arange(28), y[i*471][0:28], c='red')
        plt.title(str(i)+'_real', fontsize=40)
        plt.savefig(str(i)+'_real')
        plt.close(1)
        x_trans = trans_to_low_dim(y[i*471][0:28], embedding.weight)
        plt.figure(2, figsize=(15, 10))
        plt.plot(np.arange(num_tag_jiangwei), x_trans, c='blue')
        plt.title(str(i)+'_real' + '_trans', fontsize=40)
        plt.savefig(str(i)+'_real' + '_trans')
        plt.close(2)


