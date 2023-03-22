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

COMPUTE_LSTM_IMPORTANCE = 1
ONE_FOLD_ONLY = 1
gpu_strategy = tf.distribute.get_strategy()

with gpu_strategy.scope():
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=2021)
    test_preds = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(train, targets)):
        K.clear_session()

        print('-' * 15, '>', f'Fold {fold + 1}', '<', '-' * 15)
        X_train, X_valid = train[train_idx], train[test_idx]
        y_train, y_valid = targets[train_idx], targets[test_idx]

        # 导入已经训练好的模型
        model = keras.models.load_model('F:\sourcecode\workcode\DeeplabV3-tf2.0\models\\')
        # 计算特征重要性
        if COMPUTE_LSTM_IMPORTANCE:
            results = []
            print(' Computing LSTM feature importance...')

            for k in tqdm(range(len(COLS))):
                if k > 0:
                    save_col = X_valid[:, :, k - 1].copy()
                    np.random.shuffle(X_valid[:, :, k - 1])

                oof_preds = model.predict(X_valid, verbose=0).squeeze()
                mae = np.mean(np.abs(oof_preds - y_valid))
                results.append({'feature': COLS[k], 'mae': mae})

                if k > 0:
                    X_valid[:, :, k - 1] = save_col

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