import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LeakyReLU, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dropout, \
    BatchNormalization, LSTM, Input, concatenate, Lambda
from keras.callbacks import EarlyStopping
from data_lab import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from fft1 import *
from keras import initializers
import tensorflow as tf
from keras.utils import to_categorical
from keras.backend import expand_dims
from random import sample
from estab_day_list import *

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

path = '/data/dataDisk1/mulan/final_data3'
path2 = '/data/dataDisk1/mulan/label'


def read_day_universe_stock(date):
    path1 = '/data/remoteDir/server_200/universe/'
    codes = []
    stock_path = os.path.join(path1, str(date))
    with open(stock_path, 'r') as f:
        lines = f.readlines()
        for i in range(2000):
            line = lines[i].split()
            codes.append(line[0])
    return codes


def get_samples(path, path2, day_begin, day_end, time_step):
    day_list = estab_day_list(day_begin, day_end)
    # print(day_list)
    label_data = get_lab(path2, start=str(day_begin), end=str(day_end)).values
    codes_index1 = label_data[:, 0]  # array
    label_data = label_data[:, 1:]
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    # label_data = min_max_scaler.fit_transform(label_data.T).T
    # print(label_data)
    codes_share = {}
    codes_index2 = pd.read_csv(code_index_file, dtype={'stock_code': str}).set_index('stock_code')['idx']  # series
    # num_codes = array_data.shape[0]
    for j, code in enumerate(codes_index2.keys()):
        if code in codes_index1:
            codes_share[j] = code

    i = 0
    for date in day_list:
        path_data = path + '/' + str(date) + '.npy'
        if not os.path.exists(path_data):
            continue
        if i == 0:
            mem_data = np.load(path_data)
        else:
            mem_data = np.dstack((mem_data, np.load(path_data)))
        i += 1

    # mem_data = np.vsplit(mem_data, 3627)
    mem_data = mem_data.reshape(3627, 471, 14, mem_data.shape[2])
    num_codes = len(list(codes_share.keys()))
    X = [np.zeros((num_codes*(mem_data.shape[3] - time_step + 1 - 2), 471, 14)) for i in range(time_step)]
    Y = np.zeros((num_codes*(mem_data.shape[3] - time_step + 1 - 2), time_step, 1))
    for i0,code in enumerate(list(codes_share.keys())):
        for i in range(mem_data.shape[3] - time_step + 1 - 2):
            for j in range(time_step):
                X[j][i0*(mem_data.shape[3] - time_step + 1 - 2)+i,:,:] = mem_data[code, :,:, i+j] 
                Y[i0*(mem_data.shape[3] - time_step + 1 - 2)+i, j, 0] = label_data[int(np.argwhere(codes_index1 == codes_share[code]))][i+j+2] if not\
                    np.isnan(label_data[int(np.argwhere(codes_index1 == codes_share[code]))][i+j+2]) else 0
    np.save('/data/dataDisk1/mulan/x4', X)
    np.save('/data/dataDisk1/mulan/y4', Y)


def generate_arrays_from_files(path, path2, day_begin, day_end, time_step, batch_size=32):
    day_list = estab_day_list(day_begin, day_end)
    # print(day_list)
    label_data = get_lab(path2, start=str(day_begin), end=str(day_end)).values
    codes_index1 = label_data[:, 0]  # array
    label_data = label_data[:, 1:]
    # min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    # label_data = min_max_scaler.fit_transform(label_data.T).T
    # print(label_data)
    cnt = 0
    codes_share = {}
    codes_index2 = pd.read_csv(code_index_file, dtype={'stock_code': str}).set_index('stock_code')['idx']  # series
    # num_codes = array_data.shape[0]
    for j, code in enumerate(codes_index2.keys()):
        if code in codes_index1:
            codes_share[j] = code

    # date_list1 = []
    # for i,date in enumerate(day_list):
    #     path_data = path + '/' + str(date) + '.npy'
    #     if not os.path.exists(path_data):
    #         continue
    #     else:
    #         date_list1.append(day_list[i])
            
    i = 0
    for date in day_list:
        path_data = path + '/' + str(date) + '.npy'
        if not os.path.exists(path_data):
            continue
        if i == 0:
            mem_data = np.load(path_data)
        else:
            mem_data = np.dstack((mem_data, np.load(path_data)))
        i += 1

    mem_data = mem_data.reshape(3627, 471, 14, mem_data.shape[2])
    X = [np.zeros((batch_size, 471, 14)) for i in range(time_step)]
    Y = np.zeros((batch_size, time_step, 1))
    i0 = 0
    while True:
        code = sample(list(codes_share.keys()), 1)[0]
        for i in range(mem_data[code].shape[1] - time_step + 1 - 2):
            for j in range(time_step):
                X[j][(i0*(mem_data.shape[3] - time_step + 1 - 2)+i)%batch_size,:,:] = mem_data[code, :,:, i+j] 
                Y[(i0*(mem_data.shape[3] - time_step + 1 - 2)+i)%batch_size, j, 0] = label_data[int(np.argwhere(codes_index1 == codes_share[code]))][i+j+2] if not\
                    np.isnan(label_data[int(np.argwhere(codes_index1 == codes_share[code]))][i+j+2]) else 0
            cnt += 1
            if cnt == batch_size:
                cnt = 0
                # return (X,Y)
                yield (X, Y)
        i0 += 1


def expand_dim_backend(x):
    x1 = expand_dims(x, 1)
    return x1


class conv1D_lstm_model:
    def __init__(self):
        self.model = None

    def establish(self, day_dim, tick_len, x_dim):
        # BN = BatchNormalization(axis = 2)
        shared_conv_layer1 = Conv1D(64, 3, activation='relu', kernel_initializer='random_normal')
        shared_conv_layer2 = Conv1D(64, 3, activation='relu', kernel_initializer='random_normal')
        shared_maxpool_layer = MaxPooling1D(3)
        shared_conv_layer3 = Conv1D(32, 3, activation='relu', kernel_initializer='random_normal')
        shared_conv_layer4 = Conv1D(32, 3, activation='relu', kernel_initializer='random_normal')
        shared_avepool_layer = GlobalAveragePooling1D()
        inputa = []
        x = []
        for i in range(day_dim):
            inputa.append(Input(shape=(tick_len, x_dim)))
            # x.append(BN(inputa[i]))
            # x[i] = shared_conv_layer1(x[i])
            x.append(shared_conv_layer1(inputa[i]))
            x[i] = shared_conv_layer2(x[i])
            x[i] = shared_maxpool_layer(x[i])
            x[i] = shared_conv_layer3(x[i])
            x[i] = shared_conv_layer4(x[i])
            x[i] = shared_avepool_layer(x[i])
            x[i] = Lambda(expand_dim_backend)(x[i])
            # x[i] = BatchNormalization(axis = 2)(x[i])

        x2 = concatenate(x, axis=1)
        # print(x2)
        x2 = LSTM(32, return_sequences=True)(x2)
        y = LSTM(1, return_sequences=True)(x2)
        # x2 = Dropout(0.5)(x2)
        # y =  Dense(1, activation='sigmoid')(x2)
        self.model = Model(inputs=inputa, outputs=y)

        self.model.compile(loss='mse',
                           optimizer='rmsprop',
                           metrics=['mae'])
        print(self.model.summary())

    def train(self, generator):
        # self.model.fit(train_x, train_y, epochs=5, batch_size =  32)
        self.model.fit_generator(generator, steps_per_epoch=50, epochs=100, verbose=1, callbacks=None,
                                 validation_data=None, \
                                 validation_steps=None, class_weight=None, max_queue_size=10, workers=1,
                                 use_multiprocessing=False, initial_epoch=0)

    def predict(self, test_x):
        return self.model.predict(test_x)


if __name__ == '__main__':
    # train_x = []
    # train_y = np.random.rand(1000,5,1)
    # for i in range(5):
    #     train_x.append(np.random.rand(1000, 480, 28))
    '''
    generator = generate_arrays_from_files(path, path2, 20190301, 20190331, 5)
    conv_lstm = conv1D_lstm_model()
    conv_lstm.establish(5, 471, 14)
    conv_lstm.train(generator)
    test_x = list(np.load('/data/dataDisk1/mulan/x4.npy'))
    test_y = np.load('/data/dataDisk1/mulan/y4.npy')[:, 4, 0]
    pred_y = conv_lstm.predict(test_x)[:, 4, 0]
    p1, p2 = cal_corr(test_y, pred_y)
    print(p1, p2)
    plt.figure(1, figsize=(15, 10))
    plt.plot(np.arange(len(test_y)), test_y, c='red')   
    plt.plot(np.arange(len(pred_y)), pred_y, c='blue')
    plt.legend(labels=['real', 'predict'])
    plt.savefig(str() + '_' + str() + 'real_predict')
    plt.close(1)
    '''
    get_samples(path, path2, 20190401, 20190431, 5)
    