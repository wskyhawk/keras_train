1、train.py为训练算法



2、训练数据的格式为“C:/xxx/训练结果/xxx.png”,比如训练数据“C:/fenlei/0/abc.png”。那么0表示准确的结果，abc.png，则为切割后的图片。附件fenlei.rar为训练数据集



3、切割算法为cutimg.py，使用PIL



4、captcha_api.py是根据训练模型做了个接口提供数据接口，端口为5000