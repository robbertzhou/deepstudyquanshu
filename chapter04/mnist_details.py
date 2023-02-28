# -*- coding:utf-8 -*-
#first step :加载mnist数据集
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
#train,test数据的x/y 维度

#second EDA:对数据集进行探索和分析，首先观察训练的目标值
# print(y_train[:10])

# print(x_train[0])

#为了看清图片的数字，把非零转为1，变为黑白图片
data = x_train[0].copy()
data[data>0] = 1
text_image = []
for i in range(data.shape[0]):
    print(''.join(str(data[i])))
    # text_image.append(''.join((str(data[i]))))
    #
    # text_image.append('\n')
print(text_image)