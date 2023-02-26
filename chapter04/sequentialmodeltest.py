import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.pop()
print(f'神经网络层数：{len(model.layers)}')
model.add(tf.keras.layers.Dense(10))
print(model.layers)