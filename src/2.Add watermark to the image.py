import cv2 as cv

#图片加水印
def read_img(path):
    img = cv.imread(path,-1)

    dst = cv.resize(img,(130,124))
    ret = alpha2black_opencv2(dst)  #将水印透明部分处理为黑色

    bg = cv.imread("./lena.jpeg")   #加载背景图片
    roi = bg[300:424, 300:430]      #获取目标ROI

    mask = cv.cvtColor(ret, cv.COLOR_BGR2GRAY)  #水印和ROI叠加,变换灰度

    thresh, new_mask = cv.threshold(mask, 10, 255, cv.THRESH_BINARY)

    mask_inv = cv.bitwise_not(new_mask) #获得反向图

    new_img = cv.add(roi, 1, mask=mask_inv) #原图中抠出印章区域

    replace_img = cv.add(new_img, ret)      #将抠出印章区域得到的图片与印章相加，获取到合成图

    bg[300:424, 300:430] = replace_img      #将合成好的图片，送回到原图
    cv.imshow("bg", bg)

def alpha2black_opencv2(img):
    """
    透明部分改变为黑色
    :param img:
    :return:
    """
    width,height,channels = img.shape
    for yh in range(height):
        for xw in range(width):
            color_d = img[xw, yh]
            # 找到alpha通道为 0 的像素
            if(color_d[3] == 0):
                # 改变像素颜色为黑色
                img[xw, yh] = [0, 0, 0, 0]
    # 返回一个不透明的BGR图片
    return cv.cvtColor(img, cv.COLOR_BGRA2BGR)


if __name__ == "__main__":
    # png 透明图片路径
    path = r"./20210117190946120.png"
    read_img(path)

    cv.waitKey()
    cv.destroyAllWindows()

