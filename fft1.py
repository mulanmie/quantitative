import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import scipy.stats as stats
path = '/home/mulan/mulan/preprocessing'
path2 = '/home/mulan/Documents/fft_fig/'

def cal_corr(ticks_tag1, ticks_tag2):
    pr = stats.pearsonr(ticks_tag1, ticks_tag2)[0]
    spr = stats.spearmanr(ticks_tag1, ticks_tag2)[0]
    return pr, spr


def read_chafen_data(tag):
    file_path = path + '/' + tag + '.txt'
    file = open(file_path,'r')
    chafen_data = np.loadtxt(file,delimiter=',')
    return chafen_data

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
    tag='totbid'
    chafen_data = read_chafen_data(tag)
    # print(chafen_data[0,:4801])
    fft_for_chafen_data(tag, chafen_data)