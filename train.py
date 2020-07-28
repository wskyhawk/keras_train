"""
#使用神经网络算法训练，生成训练模型
#create by wzj
"""

import numpy as np
import os
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from PIL import Image
from sklearn.model_selection import train_test_split

def train1():
    imgpath = 'E:/img/yzm/fenlei/'

    ydict = {}
    for index, i in enumerate(list('0123456789abcdefghijklmnopqrstuvwxyz')):
        ydict[i] = index

    yresult = {}
    for i in range(len(ydict)):
        yresult[str(list(ydict.values())[i])] = list(ydict.keys())[i]

    xdata = []
    ydata = []
    for i in os.listdir(imgpath):
        dp = imgpath + os.listdir(imgpath)[ydict[i]]
        for ii in os.listdir(dp):
            xdata.append(np.array(Image.open(dp + '/' + ii)).reshape(-1))
            ydata.append(ydict[i[-1]])
        print(i, end=' ')

    x = np.array(xdata)
    y = np.array(ydata)

    x = x.reshape(-1, 25, 15, 1) / 255

    # 分割数据集
    # 随机划分样本数据为训练集和测试集
    # xtrain，ytrain：得到的训练数据。
    # xtest， ytest：得到的测试数据。
    # x, y：原始数据

    # train_test_split函数参数解析：
    # train_data：所要划分的样本特征集，指的是x
    # train_target：所要划分的样本结果，指的是y
    # test_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子，不写默认为False。

    # 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
    #     比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。不填的话默认值为False，
    #     即每次切分的比例虽然相同，但是切分的结果不同。
    #     随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：

    # 种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

    # one_hot处理
    #调用to_categorical将b按照num_classes个类别来进行转换
    ytrain = np_utils.to_categorical(ytrain, num_classes=36)
    ytest = np_utils.to_categorical(ytest, num_classes=36)


    #使用顺序式模型：通过Sequential类API
    #定义模型
    model = Sequential()

    model.add(Convolution2D(
        input_shape=(25, 15, 1),    # 输入形状就是 图片形状  # 默认 data_format:channels_last  (rows,cols,channels)
        filters=32,         #32个滤波器 －》生成32深度
        kernel_size=5,      #长度为 5
        strides=1,
        padding='same',     # 过滤模式
        activation='relu',
        name='conv1'
    ))

    model.add(MaxPooling2D(
        pool_size=2,
        strides=2,
        padding='same',
        name='pool1'
    ))
    model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling2D(2, 2, 'valid', name='pool2'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(36, activation='softmax'))

    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    model.fit(xtrain, ytrain, batch_size=64, epochs=20)

    # 评估模型
    loss, accuracy = model.evaluate(xtest, ytest)
    loss1, accuracy1 = model.evaluate(xtrain, ytrain)

    # 保存训练模型
    # model.save('wefs.h5')

    print('\ntest loss', loss)
    print('test accuracy', accuracy)

    print('\ntrain loss', loss1)
    print('train accuracy', accuracy1)

#基于现有模型再次训练
def train2():
    # imgpath = 'E:/img/yzm/fenlei/'
    imgpath = 'C:/Users/CPIC/Desktop/fenlei/'

    ydict = {}
    for index, i in enumerate(list('0123456789abcdefghijklmnopqrstuvwxyz')):
        ydict[i] = index

    yresult = {}
    for i in range(len(ydict)):
        yresult[str(list(ydict.values())[i])] = list(ydict.keys())[i]

    xdata = []
    ydata = []
    for i in os.listdir(imgpath):
        dp = imgpath + os.listdir(imgpath)[ydict[i]]
        for ii in os.listdir(dp):
            xdata.append(np.array(Image.open(dp + '/' + ii)).reshape(-1))
            ydata.append(ydict[i[-1]])
        print(i, end=' ')

    x = np.array(xdata)
    y = np.array(ydata)

    x = x.reshape(-1, 25, 15, 1) / 255

    # 分割数据集
    # 随机划分样本数据为训练集和测试集
    # xtrain，ytrain：得到的训练数据。
    # xtest， ytest：得到的测试数据。
    # x, y：原始数据

    # train_test_split函数参数解析：
    # train_data：所要划分的样本特征集，指的是x
    # train_target：所要划分的样本结果，指的是y
    # test_size：样本占比，如果是整数的话就是样本的数量
    # random_state：是随机数的种子，不写默认为False。

    # 随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
    #  比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。不填的话默认值为False，
    #  即每次切分的比例虽然相同，但是切分的结果不同。
    #  随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：

    # 种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

    # one_hot处理
    # 调用to_categorical将b按照num_classes个类别来进行转换
    ytrain = np_utils.to_categorical(ytrain, num_classes=36)
    ytest = np_utils.to_categorical(ytest, num_classes=36)

    #载入现有模型
    model=load_model('model.h5')

    # 训练模型
    # x：输入数据。如果模型只有一个输入，那么x的类型是numpy
    #  array，如果模型有多个输入，那么x的类型应当为list，list的元素是对应于各个输入的numpy array
    # y：标签，numpy array
    # batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
    # epochs=20 表示训练20轮
    model.fit(xtrain, ytrain, batch_size=64, epochs=20)

    # 评估模型
    loss, accuracy = model.evaluate(xtest, ytest)
    loss1, accuracy1 = model.evaluate(xtrain, ytrain)

    # 保存训练模型
    # model.save('wefs.h5')

    print('\ntest loss', loss)
    print('test accuracy', accuracy)

    print('\ntrain loss', loss1)
    print('train accuracy', accuracy1)

if __name__ == '__main__':
    train1() #第一次训练
    # train2() #基于现有模型进行第二次训练