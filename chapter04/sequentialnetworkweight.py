import tensorflow as tf


layer1 = tf.keras.layers.Dense(2,activation="relu",name="layer1",input_shape=(28,28))
layer2 = tf.keras.layers.Dense(3,activation="relu",name="layer2")
layer3 = tf.keras.layers.Dense(4,activation="softmax",name = "layer3")

model = tf.keras.models.Sequential(
    [layer1,layer2,layer3]
)

print(model.weights)
