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

    def train(self, generator):
        # early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        self.model.fit_generator(generator, steps_per_epoch=1, epochs=20, verbose=1, callbacks=None, validation_data=None,
                        validation_steps=None, class_weight=None, max_queue_size=10, workers=1,
                        use_multiprocessing=False, initial_epoch=0)
        # self.model.save_weights('my_model_weights.h5')
        self.weight = self.model.get_weights()[2].T

    def predict(self, test_x):
        return self.model.predict(test_x)

def generate_arrays_from_files(path, date_list, codes_index):
    tag_all = ['last', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5', 'bid_vol6',
               'bid_vol7', 'bid_vol8', 'bid_vol9','bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4',
               'ask_vol5', 'ask_vol6', 'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10', 'totoff', 'totbid',
               'amount', 'vol', 'trade']
    # tag_all = ['last', 'vol']
    num_tag = len(tag_all)
    num_codes = len(codes_index)
    num_ticks = 482
    windowsize = 11
    # list_x = np.zeros((num_codes*(num_ticks-windowsize),num_tag*(windowsize-1)))
    # list_y = np.zeros((num_codes*(num_ticks-windowsize),num_tag))
    while True:
        for day in date_list:
            read_path1 = os.path.join(path, str(day) + '_x' + '.npy')
            read_path2 = os.path.join(path, str(day) + '_y' + '.npy')
            list_x1 = np.load(read_path1)
            list_y1 = np.load(read_path2)
            yield list_x1, list_y1

def fill_zero(path1, path2, path3, date, codes_index):
    # path1 = '/data/dataDisk1/mulan/pre_data_201904/'
    # path2 = '/data/dataDisk1/mulan/final_data/'
    # path3 = '/data/dataDisk1/mulan/final_data1/'
    tag_all = ['last', 'bid1', 'ask1', 'bid_vol1', 'bid_vol2', 'bid_vol3', 'bid_vol4', 'bid_vol5', 'bid_vol6',
               'bid_vol7', 'bid_vol8', 'bid_vol9','bid_vol10', 'ask_vol1', 'ask_vol2', 'ask_vol3', 'ask_vol4',
               'ask_vol5', 'ask_vol6', 'ask_vol7', 'ask_vol8', 'ask_vol9', 'ask_vol10', 'totoff', 'totbid',
               'amount', 'vol', 'trade']
    # tag_all = ['last', 'vol']
    # stock_code_path = '/data/remoteDir/server_200/mem_data'
    # col_path = os.path.join(stock_code_path, '.index/code.csv')
    # col = pd.read_csv(col_path, dtype={'stock_code': str}).set_index('stock_code')['idx']
    # codes_index = col.index.tolist()       
    num_codes = len(codes_index)
    num_tag = len(tag_all)
    num_tag1 = 14
    num_ticks = 482
    windowsize = 11
    list_x = np.zeros((num_codes*(num_ticks-windowsize),num_tag1))
    weight = np.load(path2 + str(date_list[0]) + '-' + str(date_list[-1]) + '.npy')
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
    date_list = estab_day_list(20181201, 20181231)
    # print(date_list)
    data_path = '/data/remoteDir/server_200/mem_data/'
    path = '/data/dataDisk1/mulan/pre_data_201812/'
    path1 = '/data/dataDisk1/mulan/middle_data_201812/'
    path2 = '/data/dataDisk1/mulan/final_data/'
    path3 = '/data/dataDisk1/mulan/final_data1/'
    stock_code_path = '/data/remoteDir/server_200/mem_data'
    col_path = os.path.join(stock_code_path, '.index/code.csv')
    col = pd.read_csv(col_path, dtype={'stock_code': str}).set_index('stock_code')['idx']
    codes_index = col.index.tolist()      
    # print(codes_index)
    # codes_record, x, y = read_day_tick_data(date_list,codes_index)
    num_codes = len(codes_index)
    windowsize = 10
    num_ticks = 471
    num_tag = 28
    num_tag_jiangwei =14
    
    embedding = embedding_model()
    embedding.establish(num_tag*windowsize, num_tag)
    generator = generate_arrays_from_files(path1, date_list, codes_index)
    embedding.train(generator)
    print(embedding.weight)  
    np.save(path2 + str(date_list[0]) + '-' + str(date_list[-1]), embedding.weight) #weight(28,14)
    '''
    for date in date_list:
        print(date)
        fill_zero(path, path2, path3, date, codes_index)
    '''
