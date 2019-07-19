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

path1 = '/data/dataDisk1/mulan/work/20171030w2v_x.npy'
path2 = '/data/dataDisk1/mulan/work/20171030w2v_y.npy'

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
        self.model.add(Dense(15, input_dim=x_dim, name='dense_1'))
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


if __name__ == '__main__':
    #one_day_data = estab_one_day_data_lab(tag, day_list, is_chafen=False, is_zero_mean=False, is_minus_open=True)
    #text_like_data = estab_text_like_data_lab(one_day_data, 3)
    #x, y = generate_training_data(text_like_data, 3)
    x = np.load(path1)
    y = np.load(path2)
    embedding = embedding_model()
    embedding.establish(310, 31)
    embedding.train(x, y)
    # weight = np.zeros((4802, 32))
    # for i in range(32):
    #     weight[i,i]=1
    # print(len(weight))
    # model = embedding.model.load_weights('my_model_weights.h5')
    # np.savetxt('weight.txt',weight)
    # for i in range(10):
    #     print(i)
    #     x_orig = x[i][0:31]
    #     x_trans = trans_to_low_dim(x_orig, embedding.weight)
    #     plt.figure(1, figsize=(15, 10))
    #     plt.plot(np.arange(31), x_orig, c='blue')
    #     plt.title(str(i), fontsize=40)
    #     plt.savefig(str(i))
    #     plt.close(1)
    #     plt.figure(2, figsize=(15, 10))
    #     plt.plot(np.arange(15), x_trans, c='blue')
    #     plt.title(str(i) + '_trans', fontsize=40)
    #     plt.savefig(str(i) + '_trans')
    #     plt.close(2)
    for i in range(10):
        plt.figure(1, figsize=(15, 10))
        plt.plot(np.arange(31), y[i*470][0:31], c='red')
        plt.title(str(i)+'_real', fontsize=40)
        plt.savefig(str(i)+'_real')
        plt.close(1)
        x_trans = trans_to_low_dim(y[i*470][0:31], embedding.weight)
        plt.figure(2, figsize=(15, 10))
        plt.plot(np.arange(15), x_trans, c='blue')
        plt.title(str(i)+'_real' + '_trans', fontsize=40)
        plt.savefig(str(i)+'_real' + '_trans')
        plt.close(2)
    # y_pre = embedding.predict(x[-2:-1, :])
    # print(y_pre)
    # plt.figure(2, figsize=(15, 10))
    # plt.plot(np.arange(4802), y_pre[0, :], c='red')
    # plt.title(tag + '_' + 'pre', fontsize=40)
    # plt.savefig(tag + '_pre')
    # plt.close(2)