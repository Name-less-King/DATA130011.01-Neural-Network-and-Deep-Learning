import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import pickle


def tran_gray(image):
    # 按照灰度标准合并三通道
    image = np.array(image)
    image_gray = image[:,:,0]*0.299 + image[:,:,1]*0.587 + image[:,:,2]*0.114
    return image_gray

def truncate(image):
    # 做截断，方便后续处理
    n = 60 #100
    i,j = image.shape
    image_truncate = image[n:i-n,n:j-n]
    return image_truncate

def cut4(img):
    #切成四块，数据增强
    i, j = img.shape
    a1 = img[:i // 2, :j // 2]
    a2 = img[i // 2:, :j // 2]
    a3 = img[:i // 2, j // 2:]
    a4 = img[i // 2:, j // 2:]
    return  a1, a2, a3, a4

def cut16(img):
    #切成16块数据增强（虽然并没有用）
    a1 = img[:i//4,:j//4]
    a2 = img[i//4:2*(i//4),:j//4]
    a3 = img[2*(i//4):3*(i//4),:j//4]
    a4 = img[3*(i//4):,:j//4]
    a5 = img[:i//4,j//4:2*(j//4)]
    a6 = img[i//4:2*(i//4),j//4:2*(j//4)]
    a7= img[2*(i//4):3*(i//4),j//4:2*(j//4)]
    a8 = img[3*(i//4):,j//4:2*(j//4)]
    a9 = img[:i//4,2*(j//4):3*(j//4)]
    a10 = img[i//4:2*(i//4),2*(j//4):3*(j//4)]
    a11 = img[2*(i//4):3*(i//4),2*(j//4):3*(j//4)]
    a12 = img[3*(i//4):,2*(j//4):3*(j//4)]
    a13 = img[:i//4,3*(j//4):]
    a14 = img[i//4:2*(i//4),3*(j//4):]
    a15 = img[2*(i//4):3*(i//4),3*(j//4):]
    a16 = img[3*(i//4):,3*(j//4):]
    return a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16

if __name__=='__main__':
    # 加载图片，由于格式不同和最后测试的需要，所以需要把可疑数据，真迹和赝品分开加载

    raphael_D = []
    raphael_1 = mpimg.imread('Raphael_Project_image/1. 185x354.TIF')
    raphael_7 = mpimg.imread('Raphael_Project_image/7. 243x413 .tiff')
    raphael_10 = mpimg.imread('Raphael_Project_image/10. 269x227.TIF')
    raphael_20 = mpimg.imread('Raphael_Project_image/20. 135x390.tif')
    raphael_23 = mpimg.imread('Raphael_Project_image/23. 84x68.tif')
    raphael_25 = mpimg.imread('Raphael_Project_image/25. 156x115.tif')
    raphael_26 = mpimg.imread('Raphael_Project_image/26. 197x187.tif')
    raphael_D  = raphael_D + [raphael_1,raphael_7,raphael_10,raphael_20,raphael_23,raphael_25,raphael_26]

    raphael_T = []
    raphael_2 = mpimg.imread('Raphael_Project_image/2. 202x216.TIF')
    raphael_3 = mpimg.imread('Raphael_Project_image/3. 272x369.TIF')
    raphael_4 = mpimg.imread('Raphael_Project_image/4. 363x189.tiff')
    raphael_5 = mpimg.imread('Raphael_Project_image/5. 379x281.tiff')
    raphael_6 = mpimg.imread('Raphael_Project_image/6. 197x168.tiff')
    raphael_8 = mpimg.imread('Raphael_Project_image/8. 329x232.tif')
    raphael_9 = mpimg.imread('Raphael_Project_image/9. 279x187.tif')
    raphael_21 = mpimg.imread('Raphael_Project_image/21. 203x258.jpg')
    raphael_22 = mpimg.imread('Raphael_Project_image/22. 257x375.jpg')
    raphael_24 = mpimg.imread('Raphael_Project_image/24. 203x224.tif')
    raphael_27 = mpimg.imread('Raphael_Project_image/27. 278x419.tiff')
    raphael_T = raphael_T + [raphael_2,raphael_3,raphael_4,raphael_5,raphael_6,raphael_8,raphael_9,raphael_21,raphael_22,raphael_24,raphael_27]


    raphael_N = []
    raphael_11 = mpimg.imread('Raphael_Project_image/11. 394x248.jpg')
    raphael_12 = mpimg.imread('Raphael_Project_image/12. 305x492.jpg')
    raphael_13 = mpimg.imread('Raphael_Project_image/13. 206x184.jpg')
    raphael_14 = mpimg.imread('Raphael_Project_image/14. 279x152.jpg')
    raphael_15 = mpimg.imread('Raphael_Project_image/15. 279x152.jpg')
    raphael_16 = mpimg.imread('Raphael_Project_image/16. 335x476.jpg')
    raphael_17 = mpimg.imread('Raphael_Project_image/17. 283x300.jpg')
    raphael_18 = mpimg.imread('Raphael_Project_image/18. 217x278.jpg')
    raphael_19 = mpimg.imread('Raphael_Project_image/19. 283x191.jpg')
    raphael_N = raphael_N + [raphael_11,raphael_12,raphael_13,raphael_14,raphael_15,raphael_16,raphael_17,raphael_18,raphael_19]

    #按照reference的提示进行灰度化，便于提取线条特征
    raphael_D_2 = []
    for img in raphael_D:
        raphael_D_2 += [tran_gray(img)]
        
    raphael_N_2 =[]
    for img in raphael_N:
        raphael_N_2 += [tran_gray(img)]
        
    raphael_T_2 = []
    for img in raphael_T:
        raphael_T_2 += [tran_gray(img)]
    
    #按照reference的提示进行裁剪，因为周边通常留白，并没有笔画
    raphael_D_3 = []
    for img in raphael_D_2:
        raphael_D_3 += [truncate(img)]
    
    raphael_N_3 = []
    for img in raphael_N_2:
        raphael_N_3 += [truncate(img)]

    raphael_T_3 = []
    for img in raphael_T_2:
        raphael_T_3 += [truncate(img)]

    #17比较奇怪，左右几乎填满了 但是上下没有，需要特别裁剪
    image = raphael_N_3[6]
    i,j = image.shape
    image_trun = image[500:i-500,100:j-100]
    raphael_N_3[6] = image_trun

    # 数据扩增，切分成四个部分
    raphael_D_final = []
    for img in raphael_D_3:
        a1,a2,a3,a4 = cut4(img)
        raphael_D_final += [a1,a2,a3,a4]

    raphael_N_final = []
    for img in raphael_N_3:
        a1, a2, a3, a4 = cut4(img)
        raphael_N_final += [a1,a2,a3,a4]

    raphael_T_final = []
    for img in raphael_T_3:
        a1, a2, a3, a4 = cut4(img)
        raphael_T_final += [a1,a2,a3,a4]

    # 保存数据
    pickle.dump(raphael_N_final,open('file/raphael_N.p','wb'))
    pickle.dump(raphael_T_final,open('file/raphael_T.p','wb'))
    pickle.dump(raphael_D_final,open('file/raphael_D.p', 'wb'))
