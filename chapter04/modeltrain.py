import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']###解决中文乱码
plt.rcParams['axes.unicode_minus']=False

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5,validation_split=0.2)
# history = model.fit(x_train,y_train,epochs=5,validation_split=0.2)
# # model.evaluate(x_test,y_test)
# plt.plot(history.history['accuracy'],'r',label='正确率')
# plt.plot(history.history['val_accuracy'],'r',label='验证率')
# plt.legend()

x_train_norm,x_test_norm = x_train ,x_test
score  = model.evaluate(x_test_norm,y_test,verbose=0)
for i,x in enumerate(score):
    print(f'{model.metrics_names[i]}:{score[i]:.4f}')

predictions = model.predict(x_test_norm)
classes_x=np.argmax(predictions,axis=1)
#比对
print("actual :",y_test[0:20])
print("prediction:",classes_x[0:20])
print(f'0~9概率为：{np.around(predictions[0],2)}')
#显示第9个概率

