"""
基于训练模型启动脚本提供接口访问
"""
import numpy as np
from PIL import Image
from flask_cors import *
from flask import Flask, request, make_response, jsonify, session
from keras.models import load_model
import json
import base64
from time import time

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 解决跨域问题

@app.route('/predict', methods=['POST'])
@cross_origin()  # 解决跨域问题
# @login_required
def index():
    """
    程序启动入口
    :return:
    """
    d = json.loads(request.data.decode('utf-8'))
    imgdata = base64.b64decode(d['data'])
    path = 'F:/img/apiimg/{}.png'.format(time.time())
    with open(path, 'ab') as f:
        f.write(imgdata)
    data1 = Captcha(path).result()

    return data1


ydict = {}
for index, i in enumerate(list('0123456789abcdefghijklmnopqrstuvwxyz')):
    ydict[i] = index

yresult = {}
for i in range(len(ydict)):
    yresult[str(list(ydict.values())[i])] = list(ydict.keys())[i]


class Captcha():
    def __init__(self, path=None, model='m1.h5'):
        """
        :param path: 验证码路径
        :param model: 模型路径
        :param image: base64加密
        """
        if path:
            self.img = Image.open(path)
        self.model = load_model(model)

    def pro(self):
        a1 = self.img.convert('L')
        al = a1.load()
        for x in range(a1.size[0]):
            for y in range(a1.size[1]):
                if al[x, y] > 40 or x in [0, 1, a1.size[0] - 1] or y in [0, 1, a1.size[1] - 1]:
                    al[x, y] = 255
                else:
                    al[x, y] = 0
        return a1

    def read(self, path):
        self.img = Image.open(path)

    def cut(self):
        img = self.pro()
        a = img.crop((0, 0, 15, 25))
        b = img.crop((15, 0, 30, 25))
        c = img.crop((30, 0, 45, 25))
        d = img.crop((45, 0, 60, 25))
        if len([i for i in np.array(d).T[2:].reshape(-1) if i == 0]) < 10:
            d = img.crop((40, 0, 55, 25))
        elif len([i for i in np.array(d).T[10:].reshape(-1) if i == 0]) < 3:
            d = img.crop((42, 0, 57, 25))
        return a, b, c, d

    def img2array(self, index=0):
        img = self.cut()[index]
        array = np.array(img).reshape(-1, 25, 15, 1)
        array = array / 255
        return array

    def predict(self, array):
        try:
            r = self.model.predict(array).round()[0]
            r1 = r.tolist().index(1.0)
        except:
            r = self.model.predict(array)
            r1 = r[0].tolist().index(r[0].max())
        return yresult[str(r1)]

    def result(self):
        ra = self.predict(self.img2array(0))
        rb = self.predict(self.img2array(1))
        rc = self.predict(self.img2array(2))
        rd = self.predict(self.img2array(3))
        r = ra + rb + rc + rd
        return r


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
