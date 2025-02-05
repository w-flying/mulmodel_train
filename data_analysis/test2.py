from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.api as sm
import numpy as np

# 读取数据
df = pd.read_excel('../all_data.xlsx')
#
# # 划分自变量和因变量
# X = df[['colorfulness', 'brightness', 'quality', 'contrast', 'similarity',
#         'timelength', 'topic_complexity', 'emoji_num','likes']]
# 划分自变量和因变量
X = df[['topic_complexity', 'similarity', 'timelength', 'likes', 'OC', 'colorfulness', 'brightness', 'contrast']]
y = df['witchnum']
y=np.log(y)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

# 重置索引
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 生成平方项和交叉项
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
feature_names = poly.get_feature_names_out(X.columns)

# 只保留平方项
square_indices = [i for i, name in enumerate(feature_names) if '^2' in name]
X_train_square = X_train_poly[:, square_indices]
X_test_square = X_test_poly[:, square_indices]
square_feature_names = [feature_names[i] for i in square_indices]

# 计算特征之间的相关性
correlation_matrix = X.corr()

# 设定相关性阈值
correlation_threshold = 0.15

# 找到满足相关性阈值的特征对
selected_pairs = []
for i in range(len(X.columns)):
    for j in range(i + 1, len(X.columns)):
        if abs(correlation_matrix.iloc[i, j]) >= correlation_threshold:
            selected_pairs.append((i, j))

# 生成满足相关性阈值的交叉项
X_train_cross = []
X_test_cross = []
cross_feature_names = []
for pair in selected_pairs:
    i, j = pair
    X_train_cross.append(X_train_scaled[:, i] * X_train_scaled[:, j])
    X_test_cross.append(X_test_scaled[:, i] * X_test_scaled[:, j])
    cross_feature_names.append(f'{X.columns[i]} {X.columns[j]}')

# 将交叉项转换为 DataFrame
X_train_cross = pd.DataFrame(X_train_cross).T
X_train_cross.columns = cross_feature_names
X_test_cross = pd.DataFrame(X_test_cross).T
X_test_cross.columns = cross_feature_names

# 合并原始特征、平方项特征和交叉项特征
X_train_final = pd.concat([
    pd.DataFrame(X_train_scaled, columns=X.columns),
    pd.DataFrame(X_train_square, columns=square_feature_names),
    X_train_cross
], axis=1)
X_test_final = pd.concat([
    pd.DataFrame(X_test_scaled, columns=X.columns),
    pd.DataFrame(X_test_square, columns=square_feature_names),
    X_test_cross
], axis=1)

# 选择前 10 个与因变量相关性绝对值最大的特征
corr_with_y = X_train_final.apply(lambda x: np.corrcoef(x, y_train)[0, 1], axis=0)
corr_with_y_abs = corr_with_y.abs()
top_10_features = corr_with_y_abs.nlargest(14, keep='all').index

X_train_top10 = X_train_final[top_10_features]
X_test_top10 = X_test_final[top_10_features]

# 定义一个函数来评估负二项模型
def evaluate_negative_binomial_model(X_train, y_train, X_test, y_test):
    # 添加常数项
    X_train_with_const = sm.add_constant(X_train)
    X_test_with_const = sm.add_constant(X_test)

    model = sm.NegativeBinomial(y_train, X_train_with_const)
    result = model.fit(method='bfgs', tol=1e-8, maxiter=1000)
    y_pred = result.predict(X_test_with_const)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    p_values = result.pvalues  # 获取每个系数的 p 值
    return mse, r2, result, p_values

def train_final_model(X_train, y_train, X_test, y_test, feature_names):
    mse, r2, result, p_values = evaluate_negative_binomial_model(X_train, y_train, X_test, y_test)
    # 输出变量和对应的参数
    print('\n\n最终的模型信息')
    print(f"均方误差 (MSE): {mse}")
    print(f"决定系数 (R^2): {r2}")
    # 获取特征名称
    feature_names_with_const = ['const'] + list(feature_names)
    # 获取模型系数
    coefficients = result.params
    fea_p = []
    # 写入每个自变量对应的参数值和 p 值
    for feature, coef, p_value in zip(feature_names_with_const, coefficients, p_values):
        fea_p.append([feature, coef, p_value])
    d_temp = pd.DataFrame(fea_p, columns=['feature', 'coef', 'p_value'])
    d_temp['p_value_*'] = d_temp['p_value'].apply(charge_p)
    d_temp.to_excel('feature_coef_p.xlsx', index=False)

def charge_p(p):
    if p < 0.001:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return 'False'

# 训练最终模型
train_final_model(X_train_top10, y_train, X_test_top10, y_test, X_train_top10.columns)