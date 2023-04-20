# 导入相关模块
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 导入iris数据集
iris = load_iris()
data, target = iris.data, iris.target
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=43)
# 创建lightgbm分类模型
gbm = lgb.LGBMClassifier(objective='multiclass',
                         num_class=3,
                         num_leaves=31,
                         learning_rate=0.05,
                         n_estimators=20)
# 模型训练
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=5)
# 预测测试集
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型评估
print('Accuracy of lightgbm:', accuracy_score(y_test, y_pred))
# 绘制模型特征重要性
lgb.plot_importance(gbm)
plt.show();