import cv2
import pickle
from sklearn.model_selection import train_test_split
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.prepare_data import *


def ft_img(img):
    ft_img = []
    for k in range(24):
        res = cv2.filter2D(img, -1, gabor_filter[k])
        ft_img += [res]
        print(k)
        del res
    return ft_img

def stat_1(img):
    # 均值
    return img.mean()


def stat_2(img):
    # 方差
    return np.std(img)


def stat_3(img):
    # 尾部百分比
    n, m = img.shape
    mean = stat_1(img)
    std = stat_2(img)
    count = sum(sum(abs(img - mean) > std))
    return count / (m * n)

def stat_4(img):
    # 计算能量
    return sum(sum(img**2))

def gabor_fn(sigma, alpha, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    max = 30
    min = -max
    (y, x) = np.meshgrid(np.arange(min, max + 1), np.arange(min, max + 1))

    # Rotation
    x_alpha = x * np.cos(alpha) + y * np.sin(alpha)
    y_alpha = -x * np.sin(alpha) + y * np.cos(alpha)

    gb = np.exp(-.5 * (x ** 2 / sigma_x ** 2 + y ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_alpha + psi)
    return gb

def feature_set(S):
    for k in range(len(S)):
        img = S[k]
        ft = ft_img(img)
        single_S = []
        for loop in range(len(ft)):
            st_1 = stat_1(ft[loop])
            st_2 = stat_2(ft[loop])
            st_3 = stat_3(ft[loop])
            st_4 = stat_4(ft[loop])
            single_S += [st_1, st_2, st_3,st_4]
        index = [2, 3, 4, 5, 6, 8, 9, 21, 22, 24, 27] # for T
        # index = [1,7,10,20,23,25,26] # for D
        # index = [11, 12, 13, 14, 15, 16, 17, 18, 19]  # for N
        pickle.dump(single_S, open('file/N_' + str(index[k//4]) + '_'+str(k%4+1) +'.p', 'wb'))

if __name__ == '__main__':
    # 读取灰度图
    raphael_D = pickle.load(open('file/raphael_D.p', 'rb'))
    raphael_T = pickle.load(open('file/raphael_T.p', 'rb'))
    raphael_N = pickle.load(open('file/raphael_N.p', 'rb'))
    '''
    #设置 gabor 过滤器 提取特征，之后注释掉，避免反复提取
    sigma = 10
    psi = 0
    gamma = 1
    gabor_filter = []
    for i in range(6):
        for j in range(4):
            alpha = i * np.pi/6
            Lambda = 5*(j+1)
            gabor_filter += [np.array(gabor_fn(sigma, alpha, Lambda, psi, gamma))]
    
    """extracting 54 feature for every painting set: N,T,D"""
    feature_set(raphael_T)
    # feature_set(raphael_D)
    #feature_set(raphael_N)
    '''

    tight_frame_data = []
    path = 'file'
    Files_name = os.listdir(path)
    for file in Files_name: 
        if not os.path.isdir(file):
            if (file != '.DS_Store'):
                with open(path + "/" + file, 'rb') as load_file:
                    print(file)
                    a=pickle.load(load_file)
                    tight_frame_data += [a]

    D = tight_frame_data[0:len(raphael_D)]
    N = tight_frame_data[len(raphael_D):len(raphael_N)+len(raphael_D)]
    T = tight_frame_data[len(raphael_N)+len(raphael_D):len(raphael_N)+len(raphael_D)+len(raphael_T)]
    pickle.dump(np.array(T), open('data/tight_frame_T.p', 'wb'))
    pickle.dump(np.array(N), open('data/tight_frame_N.p', 'wb'))
    pickle.dump(np.array(D), open('data/tight_frame_D.p', 'wb'))
