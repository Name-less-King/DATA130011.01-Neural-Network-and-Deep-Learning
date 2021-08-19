from matplotlib import pyplot
from utils.prepare_data import *
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import os



def ft_img(img):
    ft_img = []
    for k in range(24):
        res = cv2.filter2D(img, -1, tau[k])
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


def feature_set(S):
    for k in range(len(S)):
        img = S[k]
        ft = ft_img(img)
        single_S = []
        for loop in range(len(ft)):
            st_1 = stat_1(ft[loop])
            st_2 = stat_2(ft[loop])
            st_3 = stat_3(ft[loop])
            single_S += [st_1, st_2, st_3]

        # index = [2, 3, 4, 5, 6, 8, 9, 21, 22, 24, 27] # for T
        # index = [1,7,10,20,23,25,26] # for D
        index = [11, 12, 13, 14, 15, 16, 17, 18, 19]  # for N
        pickle.dump(single_S, open('data/N_' + str(index[k//4]) + '_'+str(k%4+1) +'.p', 'wb'))


if __name__ == '__main__':
    # 加载灰度图
    raphael_D = pickle.load(open('data/raphael_D.p', 'rb'))
    raphael_T = pickle.load(open('data/raphael_T.p', 'rb'))
    raphael_N = pickle.load(open('data/raphael_N.p', 'rb'))


    '''
    # 论文中提到的18个过滤器 提取特征后注释掉，避免浪费计算资源
    tau = []
    tau_0 = np.array([[1,2,1],      [2,4,2],    [1,2,1]])/16
    tau_1 = np.array([[1,0,-1],     [2,0,-2],   [1,0,-1]])/16
    tau_2 = np.array([[1,2,1],      [0,0,0],    [-1,-2,-1]])/16
    tau_3 = np.array([[1,1,0],      [1,0,-1],   [0,-1,-1]])*(2**0.5)/16
    tau_4 = np.array([[0,1,1],      [-1,0,1],   [-1,-1,0]])*(2**0.5)/16
    tau_5 = np.array([[1,0,-1],     [0,0,0],    [-1,0,1]])*(7**0.5)/24
    tau_6 = np.array([[-1,2,-1],    [-2,4,-2],  [-1,2,-1]])/48
    tau_7 = np.array([[-1,2,-1],    [2,4,2],    [-1,-2,-1]])/48
    tau_8 = np.array([[0,0,-1],     [0,2,0],    [-1,0,0]])/12
    tau_9 = np.array([[-1,0,0],     [0,2,0],    [0,0,-1]])/12
    tau_10 = np.array([[0,1,0],     [-1,0,-1],  [0,1,0]])*(2**0.5)/12
    tau_11 = np.array([[-1,0,1],    [2,0,-2],   [-1,0,1]])*(2**0.5)/16
    tau_12 = np.array([[-1,2,-1],   [0,0,0],    [1,-2,1]])*(2**0.5)/16
    tau_13 = np.array([[1,-2,1],    [-2,4,-2],  [1,-2,1]])/48
    tau_14 = np.array([[0,0,0],     [-1,2,-1],  [0,0,0]])*(2**0.5)/12
    tau_15 = np.array([[-1,2,-1],   [0,0,0],    [-1,2,-1]])*(2**0.5)/24
    tau_16 = np.array([[0,-1,0],    [0,2,0],    [0,-1,0]])*(2**0.5)/12
    tau_17 = np.array([[-1,0,-1],   [2,0,2],    [-1,0,-1]])*(2**0.5)/24
    tau += [tau_0,tau_1,tau_2,tau_3,tau_4,tau_5,tau_6,tau_7,tau_8,tau_9,tau_10,tau_11,tau_12,tau_13,tau_14,tau_15,tau_16,tau_17]
    pickle.dump(tau,open('tau.p','wb'))
    '''
    tau = pickle.load(open('tau.p', 'rb'))

    feature_set(raphael_T)
    feature_set(raphael_D)
    feature_set(raphael_N)

    tight_frame_data = []
    path = 'data'
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




