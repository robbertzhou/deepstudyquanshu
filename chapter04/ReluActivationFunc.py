import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,21)
x_tf = tf.constant(x,dtype=tf.float32)

y = tf.keras.activations.relu(x_tf,threshold=5,max_value=9).numpy()

plt.plot(x,y)
plt.show()