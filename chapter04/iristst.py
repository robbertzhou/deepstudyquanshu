import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
data = load_iris()
iris_data = np.float32(data.data)
iris_target = data.target
iris_target = np.float32(tf.keras.utils.to_categorical(iris_target,num_classes=3))
train_data = tf.data.Dataset.from_tensor_slices((iris_data,iris_target)).batch(128)
input_xs = tf.keras.Input(shape=(4,),name='input_xs')
out = tf.keras.layers.Dense(32,activation='relu')(input_xs)
out = tf.keras.layers.Dense(64,activation='relu')(out)
logits = tf.keras.layers.Dense(3,activation='softmax')(out)
model = tf.keras.Model(inputs = input_xs,outputs = logits)
opt = tf.optimizers.Adam(1e-3)
model.compile(optimizer=opt,loss = tf.losses.categorical_crossentropy,metrics=['accuracy'])
model.fit(train_data,epochs=500)
score = model.evaluate(iris_data,iris_target)
model.save("iris.h5")
# 150	4	setosa	versicolor	virginica


import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import RobustScaler, normalize
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from IPython.display import display
import pandas as pd

COMPUTE_LSTM_IMPORTANCE = 1
ONE_FOLD_ONLY = 1
gpu_strategy = tf.distribute.get_strategy()

EPOCH = 300
BATCH_SIZE = 1024
NUM_FOLDS = 10

with gpu_strategy.scope():
    kf = KFold(n_splits=2, shuffle=True, random_state=2021)
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(iris_data, iris_target)):
        K.clear_session()

        print('-' * 15, '>', f'Fold {fold + 1}', '<', '-' * 15)
        X_train, X_valid = iris_data[train_idx], iris_data[test_idx]
        y_train, y_valid = iris_target[train_idx], iris_target[test_idx]

        # 导入已经训练好的模型
        model = keras.models.load_model('iris.h5')
        COLS = ["150",	"4",	"setosa",	"versicolor",	"virginica"]
        # 计算特征重要性
        if COMPUTE_LSTM_IMPORTANCE:
            results = []
            print(' Computing LSTM feature importance...')

            for k in tqdm(range(len(COLS))):
                if k > 0:
                    save_col = X_valid[:, k - 1].copy()
                    np.random.shuffle(X_valid[:, k - 1])

                oof_preds = model.predict(X_valid, verbose=0).squeeze()
                mae = np.mean(np.abs(oof_preds - y_valid))
                results.append({'feature': COLS[k], 'mae': mae})

                if k > 0:
                    X_valid[:, k - 1] = save_col

            # 展示特征重要性
            print()
            df = pd.DataFrame(results)
            df = df.sort_values('mae')
            plt.figure(figsize=(10, 20))
            plt.barh(np.arange(len(COLS)), df.mae)
            plt.yticks(np.arange(len(COLS)), df.feature.values)
            plt.title('LSTM Feature Importance', size=16)
            plt.ylim((-1, len(COLS)))
            plt.show()

            # SAVE LSTM FEATURE IMPORTANCE
            df = df.sort_values('mae', ascending=False)
            df.to_csv(f'lstm_feature_importance_fold_{fold}.csv', index=False)

        # ONLY DO ONE FOLD
        if ONE_FOLD_ONLY: break