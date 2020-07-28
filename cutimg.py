"""
使用PIL切割验证码图片
"""
from PIL import Image
import numpy as np
import cv2
import os

class cutimg:
    def __init__(self):
        print('通过PIL切割验证码图片')

    def smartSliceImg(self, img, outDir, count=4, p_w=3):
        """
        比较正常的验证码可以通过这个来进行切割
        :param img:
        :param outDir:
        :param count: 图片中有多少个图片
        :param p_w: 对切割地方多少像素进行判断
        :return:
        """
        w, h = img.size
        pixdata = img.load()
        eachWidth = int(w / count)
        beforeX = 0

        for i in range(count):
            allBCount = []
            nextXOri = (i + 1) * eachWidth

            for x in range(nextXOri - p_w, nextXOri + p_w):
                if x >= w:
                    x = w - 1
                if x < 0:
                    x = 0
                b_count = 0

                for y in range(h):
                    if pixdata[x, y] == 0:
                        b_count += 1
                allBCount.append({'x_pos': x, 'count': b_count})

            sort = sorted(allBCount, key=lambda e: e.get('count'))

            nextX = sort[0]['x_pos']
            box = (beforeX, 0, nextX, h)

            img.crop(box).save(outDir + str(i) + '.png')
            beforeX = nextX

    def getPoint(self,x, y, data, subdata=None):
        a = [0, -1, 0, 1, 0, -2, 0, 2, 0, -3, 0, 3, 0, -4, 0, 4, 0, -5, 0, 5]
        b = [1, 0, -1, 0, 2, 0, -2, 0, 3, 0, -3, 0, 4, 0, -4, 0, 5, 0, -5, 0]
        width, height = data.shape
        if subdata is None:
            subdata = []
        if x > 5 and y < height - 5 and y > 5 and x < width - 5:
            for i in range(20):
                if data[x + a[i]][y + b[i]] == 1:
                    subdata.append((x + a[i], y + b[i]))
                    data[x + a[i]][y + b[i]] = 2
                    self.getPoint(x + a[i], y + b[i], data, subdata)
        subdata.append((x, y))

    def getcell(self,data):
        list1 = []
        index = 0
        flag = True
        for y in range(data.shape[1]):
            for x in range(data.shape[0]):
                if data[x][y] == 1:
                    if list1:
                        for i in range(len(list1)):
                            if (x, y) in list1[i]:
                                flag = False
                    if not flag:
                        continue
                    list1.append([])
                    self.getPoint(x, y, data, list1[index])  # 调用流水算法
                    index += 1
                else:
                    continue

        for index in range(len(list1)):
            l = list1[index][0][0]
            t = list1[index][0][1]
            r = list1[index][0][0]
            b = list1[index][0][1]
            for i in list1[index]:
                x = i[0]
                y = i[1]
                l = min(l, x)
                t = min(t, y)
                r = max(r, x)
                b = max(b, y)
            w = r - l + 1
            h = b - t + 1
            if (w * h < 8):  # 去除小色块
                continue
            img0 = np.zeros([w, h])  # 创建全0矩阵
            for x, y in list1[index]:
                img0[x - l][y - t] = 1
            img0[img0 < 1] = 255
            img1 = Image.fromarray(img0)
            img1 = img1.convert('RGB')
            img1.save('C:\\Users\\Desktop\\code\\222\\' + str(index) + '.png')


    def splitimage(self, rownum=1, colnum=4, dstpath="C:\\Users\\Desktop\\code\\222\\",
                   img_name="C:\\Users\\Desktop\\code\\545.png"):
        img = Image.open(img_name)
        w, h = img.size
        if rownum <= h and colnum <= w:
            print('Original image info: %sx%s, %s, %s' % (w, h, img.format, img.mode))
            print('开始处理图片切割, 请稍候...')

            s = os.path.split(img_name)
            if dstpath == '':
                dstpath = s[0]
            fn = s[1].split('.')
            basename = fn[0]
            ext = fn[-1]

            num = 1
            rowheight = h // rownum
            colwidth = w // colnum
            file_list = []
            for r in range(rownum):
                index = 0
                for c in range(colnum):
                    # (left, upper, right, lower)
                    # box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                    if index < 1:
                        colwid = colwidth + 6
                    elif index < 2:
                        colwid = colwidth + 1
                    elif index < 3:
                        colwid = colwidth

                    box = (c * colwid, r * rowheight, (c + 1) * colwid, (r + 1) * rowheight)
                    newfile = os.path.join(dstpath, basename + '_' + str(num) + '.' + ext)
                    file_list.append(newfile)
                    img.crop(box).save(os.path.join(dstpath, basename + '_' + str(num) + '.' + ext), ext)
                    num = num + 1
                    index += 1
            for f in file_list:
                print(f)
            print('图片切割完毕，共生成 %s 张小图片。' % num)


if __name__ == '__main__':
    img = Image.open('C:\\Users\\Desktop\\code\\545.png')
    img.convert('L')
    outDir = 'C:\\Users\\Desktop\\code\\111\\'

    cutimg2 = cutimg()
    cutimg2.smartSliceImg(img, outDir, count=4, p_w=1)

    #
    #filename = 'C:\\Users\\Desktop\\code\\2.png'
    # data = cv2.imread(filename, 2)
    # allimg = cutimg2.getcell(data)

    cutimg2.splitimage(rownum=1, colnum=4, dstpath="C:\\Users\\Desktop\\code\\222\\",
                   img_name="C:\\Users\\Desktop\\code\\545.png")