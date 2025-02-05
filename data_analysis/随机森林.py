import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 设置 Matplotlib 的字体为支持中文的字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体，如黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 数据加载与预处理
data = pd.read_excel("../all_data.xlsx")

# 更谨慎的特征筛选（保留可能有价值的特征）
cols_to_drop = ["text", "img_name", "title_x", "title_y", "img_url", "img_load"]
data_clean = data.drop(columns=cols_to_drop)

# 数据探索
print("缺失值统计：")
print(data_clean.isnull().sum())
print("\n数据描述：")
data_clean.describe().to_excel("data_clean_description.xlsx")
print(data_clean.describe())

# 处理缺失值和异常值
# 删除缺失值
data_clean = data_clean.dropna()

# 筛选witchnum大于0且小于第95百分位数的数据
data_clean = data_clean[(data_clean["witchnum"] > 0) & (data_clean["witchnum"] < data_clean["witchnum"].quantile(0.95))]

# 目标变量转换验证
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data_clean["witchnum"], kde=True)
plt.title("原始目标变量分布")

plt.subplot(1, 2, 2)
y_log = np.log1p(data_clean["witchnum"])  # 使用log1p避免log(0)
sns.histplot(y_log, kde=True)
plt.title("对数转换后分布")
plt.tight_layout()
plt.show()
plt.savefig("witchnum_distribution.png")
# 特征工程
X = data_clean.drop(columns=["witchnum"])
y = y_log  # 使用已验证的对数转换


# 创建随机森林回归模型
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
# 训练模型
base_model.fit(X, y)
# 计算特征重要性得分
importances = base_model.feature_importances_
# 按重要性得分排序
sorted_indices = np.argsort(importances)[::-1]
# 计算前 80% 的特征数量
num_features = int(len(importances) * 0.8)
# 获取前 80% 特征的最小重要性得分作为阈值
threshold = importances[sorted_indices[num_features - 1]]

selector = SelectFromModel(base_model, prefit=True, threshold=threshold)
# 对特征进行选择
X_selected = selector.transform(X)
# 获取选择的特征
selected_features = X.columns[selector.get_support()]

print("\n筛选后的特征：", selected_features.tolist())
# 数据集划分（分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=0.2,
    random_state=42,
    stratify=pd.qcut(y, q=5)  # 基于目标变量分层
)

# 模型调优
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 0.8]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\n最佳参数：", grid_search.best_params_)
print("最佳验证R²：", grid_search.best_score_)

# 交叉验证评估
cv_scores = cross_val_score(best_model, X_selected, y, cv=5, scoring='r2')
print("\n交叉验证R²：", cv_scores)
print("平均交叉验证R²：", np.mean(cv_scores))

# 最终评估
y_pred = best_model.predict(X_test)

# 将预测值转换回原始尺度
y_test_exp = np.expm1(y_test)
y_pred_exp = np.expm1(y_pred)

print("\n测试集指标：")
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test_exp, y_pred_exp))
print("RMSE:", np.sqrt(mean_squared_error(y_test_exp, y_pred_exp)))

# 特征重要性可视化
importance = pd.Series(best_model.feature_importances_, index=selected_features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importance.plot(kind='barh')
plt.title("特征重要性")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
plt.savefig("特征重要性.png")

# 残差分析
residuals = y_test_exp - y_pred_exp
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_exp, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("残差图")
plt.xlabel("预测值")
plt.ylabel("残差")
plt.tight_layout()
plt.show()
plt.savefig("残差图.png")