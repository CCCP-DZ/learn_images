
import numpy as np
import cv2
import matplotlib.pyplot as plt

def IdealHighPassFiltering(f_shift):   #高斯高通滤波,
    D0 = 30# 设置滤波半径
    # 初始化
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m/2)
    y0 = np.floor(n/2)
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - x0)**2 + (j - y0)**2)
            if D >= D0:
                h1[i][j] = 1
    result = np.multiply(f_shift, h1)#数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
    return result

def GaussLowPassFiltering(f_shift): #高斯低通滤波
    D0 = 30# 设置滤波半径
    # 初始化
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m/2)
    y0 = np.floor(n/2)
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - x0)**2 + (j - y0)**2)
            h1[i][j] = np.exp((-1)*D**2/2/(D0**2))
    result = np.multiply(f_shift, h1)#数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
    return result

def butter_worth_LowPassFiltering(f_shift): #巴特沃斯低通滤波器
    p=2#参数赋初始值
    d0 = 30
    # 初始化
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m/2)
    y0 = np.floor(n/2)
    for i in range(m):
        for j in range(n):
            D = np.sqrt((i - x0)**2 + (j - y0)**2)
            h1[i][j] = 1 / (1 + 0.414*(D / d0) ** (2 * p))#计算传递函数
    result = np.multiply(f_shift, h1)#数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
    return result

def butter_worth_HighPassFiltering(f_shift): #巴特沃斯高通滤波器
    p=2#参数赋初始值
    d0 = 30
    # 初始化
    m = f_shift.shape[0]
    n = f_shift.shape[1]
    h1 = np.zeros((m, n))
    x0 = np.floor(m/2)
    y0 = np.floor(n/2)
    for i in range(m):
        for j in range(n):

            D = np.sqrt((i - x0)**2 + (j - y0)**2)

            if D==0:
                h1[i][j]=0
            else:
                h1[i][j] = 1 / (1 + 0.414* (D / d0) **(2 * p))#计算传递函数

    result = np.multiply(f_shift, h1)#数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
    return result

img =cv2.imread('demo.png',0)
f=np.fft.fft2(img)#计算二维的傅里叶变换
f_shift=np.fft.fftshift(f)#将图像中的低频部分移动到图像的中心

img1 =cv2.imread('test.jpg',0)
f1=np.fft.fft2(img1)##计算二维的傅里叶变换
f_shift1=np.fft.fftshift(f1)#将图像中的低频部分移动到图像的中心

# 显示图像

plt.figure(figsize=(17,10),facecolor='green')#创建figure
plt.rcParams['font.sans-serif']='STXINWEI'#设置中文字体为华文新魏


# 高斯低通滤波
GLPF = GaussLowPassFiltering(f_shift)
new_f1 = np.fft.ifftshift(GLPF)#进图像的低频和高频部分移动到图像原来的位置
new_image1 = np.uint8(np.real(np.fft.ifft2(new_f1)))#先将移频后的信号还原成之前的然后对复数进行操作，返回复数类型参数的实部，

#巴特沃斯低通滤波
BLPF = butter_worth_LowPassFiltering(f_shift)
new_f2 = np.fft.ifftshift(BLPF)#进图像的低频和高频部分移动到图像原来的位置
new_image2 = np.uint8(np.real(np.fft.ifft2(new_f2)))#先将移频后的信号还原成之前的然后对复数进行操作，返回复数类型参数的实部，

#巴特沃斯高通滤波
BHPF = butter_worth_HighPassFiltering(f_shift1)
new_f3 = np.fft.ifftshift(BHPF)#进图像的低频和高频部分移动到图像原来的位置
new_image3 = np.uint8(np.real(np.fft.ifft2(new_f3)))#先将移频后的信号还原成之前的然后对复数进行操作，返回复数类型参数的实部，


# 理想高斯高通滤波
IHPF = IdealHighPassFiltering(f_shift1)
new_f4 = np.fft.ifftshift(IHPF)#进图像的低频和高频部分移动到图像原来的位置
new_image4 = np.uint8(np.abs(np.fft.ifft2(new_f4)))#先将移频后的信号还原成之前的然后对复数进行操作，返回复数类型参数的实部，


titles = ['原始图像', '原始图像', '原始图像', '原始图像','高斯低通滤波图像','巴特沃斯低通滤波图像','巴特沃斯高通滤波图像','高斯高通滤波图像']
images = [img, img, img1, img1,new_image1,new_image2,new_image3,new_image4]

for i in range(8):
    plt.subplot(2, 4, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontsize=15)

plt.show()
