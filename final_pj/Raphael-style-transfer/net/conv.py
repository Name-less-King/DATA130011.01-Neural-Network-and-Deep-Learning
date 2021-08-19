import numpy as np
from scipy.signal import convolve2d

def padding(x,P):
    y = np.pad(x,((P,P),(P,P)))
    return y

def point_conv(a, W):
    z = np.sum(np.sum(a * W))
    return z

def convolution(x,Kernal,P=None,S=None):
    #输入原始图片x，卷积核K，P表示填充参数，S表示卷积步长，输出卷积后的图片y
    H, W = x.shape
    K, _ = Kernal.shape
    #默认卷积后图像大小不变
    if S == None:
        S = 1
    if P == None:
        P = K // 2
	
    #计算输出图片的大小
    Hout =  (H - K + 2*P) // S + 1
    Wout =  (W - K + 2*P) // S + 1

    y = np.zeros((Hout,Wout))
    xpad = padding(x,P)
                                               
    for h in range(Hout):                           
        for w in range(Wout):                                     
            up = h * S         
            down = up + K      
            left = w * S        
            right = left + K     
            xwindow = xpad[up:down, left:right]
            y[h, w] = point_conv(xwindow, Kernal)
    return y

#定义18个卷积核
filters = [
    np.dot(1/16, [1, 2, 1, 2, 4, 2, 1, 2, 1]).reshape(3, 3),
    np.dot(1/16, [1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3),
    np.dot(1/16, [1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape(3, 3),
    np.dot(np.sqrt(2)/16, [1, 1, 0, 1, 0, -1, 0, -1, -1]).reshape(3, 3),
    np.dot(np.sqrt(2)/16, [0, 1, 1, -1, 0, 1, -1, -1, 0]).reshape(3, 3),
    np.dot(np.sqrt(7)/24, [1, 0, -1, 0, 0, 0, -1, 0, 1]).reshape(3, 3),
    np.dot(1/48, [-1, 2, -1, -2, 4, -2, -1, 2, -1]).reshape(3, 3),
    np.dot(1/48, [-1, -2, -1, 2, 4, 2, -1, -2, -1]).reshape(3, 3),
    np.dot(1/12, [0, 0, -1, 0, 2, 0, -1, 0, 0]).reshape(3, 3),
    np.dot(1/12, [-1, 0, 0, 0, 2, 0, 0, 0, -1]).reshape(3, 3),
    np.dot(np.sqrt(2)/12, [0, 1, 0, -1, 0, -1, 0, 1, 0]).reshape(3, 3),
    np.dot(np.sqrt(2)/16, [-1, 0, 1, 2, 0, -2, -1, 0, 1]).reshape(3, 3),
    np.dot(np.sqrt(2)/16, [-1, 2, -1, 0, 0, 0, 1, -2, 1]).reshape(3, 3),
    np.dot(1/48, [1, -2, 1, -2, 4, -2, 1, -2, 1]).reshape(3, 3),
    np.dot(np.sqrt(2)/12, [0, 0, 0, -1, 2, -1, 0, 0, 0]).reshape(3, 3),
    np.dot(np.sqrt(2)/24, [-1, 2, -1, 0, 0, 0, -1, 2, -1]).reshape(3, 3),
    np.dot(np.sqrt(2)/12, [0, -1, 0, 0, 2, 0, 0, -1, 0]).reshape(3, 3),
    np.dot(np.sqrt(2)/24, [-1, 0, -1, 2, 0, 2, -1, 0, -1]).reshape(3, 3),
]

def extract_features(x,Kernal):
    y = convolve2d(x,Kernal,'same')
    H,W  = y.shape
    mean = np.sum(y) / (H*W)
    std = np.sum(y**2 - mean) / (H*W-1)
    tail = np.sum(np.abs(y-mean) > std) / (H*W)

    return mean,std,tail 

def extract_all_features(x):
    C,H,W = x.shape
    gray = np.zeros((H,W))
    gray=0.299*x[0]+0.587*x[1]+0.114*x[2]
    features = []

    for k in filters:
        mean,std,tail = extract_features(gray,k)
        features.append(mean)
        features.append(std)
        features.append(tail)
    return gray,features
