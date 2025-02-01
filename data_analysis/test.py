from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import statsmodels.api as sm
import warnings

warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_excel('../all_data.xlsx')

# 划分自变量和因变量
X = df[['EC', 'OC', 'colorfulness', 'brightness', 'quality','contrast','similarity',
        'timelength', 'topic_complexity', 'emoji_num','likes']]
y = df['witchnum']
# y值除10000
y = y / 10000

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
correlation_threshold = 1.0

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

# 将交叉项转换为数组
X_train_cross = pd.DataFrame(X_train_cross).T
X_test_cross = pd.DataFrame(X_test_cross).T

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

# 定义一个函数来评估负二项模型
def evaluate_negative_binomial_model(X_train, y_train, X_test, y_test):
    # 添加常数项
    X_train_with_const = sm.add_constant(X_train)
    X_test_with_const = sm.add_constant(X_test)

    model = sm.NegativeBinomial(y_train, X_train_with_const)
    result = model.fit()
    y_pred = result.predict(X_test_with_const)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    p_values = result.pvalues  # 获取每个系数的 p 值
    return mse, r2, result, p_values

# 向前选择变量
def forward_selection(X_train, y_train, X_test, y_test, feature_names):
    selected_features = []
    remaining_features = list(range(X_train.shape[1]))
    best_r2 = -float('inf')
    while remaining_features:
        best_feature = None
        for feature in remaining_features:
            temp_features = selected_features + [feature]
            X_train_subset = X_train[:, temp_features]
            X_test_subset = X_test[:, temp_features]
            _, r2, _, _ = evaluate_negative_binomial_model(X_train_subset, y_train, X_test_subset, y_test)
            if r2 >= best_r2:
                best_r2 = r2
                best_feature = feature
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break
    return [feature_names[i] for i in selected_features]

# 向后选择变量
def backward_selection(X_train, y_train, X_test, y_test, feature_names):
    selected_features = list(range(X_train.shape[1]))
    _, best_r2, _, _ = evaluate_negative_binomial_model(X_train, y_train, X_test, y_test)
    while len(selected_features) > 1:
        worst_feature = None
        for feature in selected_features:
            temp_features = selected_features.copy()
            temp_features.remove(feature)
            X_train_subset = X_train[:, temp_features]
            X_test_subset = X_test[:, temp_features]
            _, r2, _, _ = evaluate_negative_binomial_model(X_train_subset, y_train, X_test_subset, y_test)
            if r2 >= best_r2:
                best_r2 = r2
                worst_feature = feature
        if worst_feature is not None:
            selected_features.remove(worst_feature)
        else:
            break
    return [feature_names[i] for i in selected_features]

def forward_selection_model(X_train, y_train, X_test, y_test, feature_names):
    # 执行向前选择
    forward_selected_features = forward_selection(X_train, y_train, X_test, y_test, feature_names)
    print("向前选择的特征:", forward_selected_features)
    # 使用向前选择的特征重新训练模型
    forward_selected_indices = [list(feature_names).index(f) for f in forward_selected_features]
    X_train_forward = X_train[:, forward_selected_indices]
    X_test_forward = X_test[:, forward_selected_indices]
    mse_forward, r2_forward, result_forward, p_values_forward = evaluate_negative_binomial_model(X_train_forward, y_train, X_test_forward, y_test)
    print(f"向前选择后模型的均方误差 (MSE): {mse_forward}")
    print(f"向前选择后模型的决定系数 (R^2): {r2_forward}")
    return [X_train_forward, y_train, X_test_forward, y_test, result_forward, p_values_forward, forward_selected_indices]

def backward_selection_model(X_train, y_train, X_test, y_test, feature_names):
    # 执行向后选择
    backward_selected_features = backward_selection(X_train, y_train, X_test, y_test, feature_names)
    print("向后选择的特征:", backward_selected_features)
    # 使用向后选择的特征重新训练模型
    backward_selected_indices = [list(feature_names).index(f) for f in backward_selected_features]
    X_train_backward = X_train[:, backward_selected_indices]
    X_test_backward = X_test[:, backward_selected_indices]
    mse_backward, r2_backward, result_backward, p_values_backward = evaluate_negative_binomial_model(X_train_backward, y_train, X_test_backward, y_test)
    print(f"向后选择后模型的均方误差 (MSE): {mse_backward}")
    print(f"向后选择后模型的决定系数 (R^2): {r2_backward}")
    return [X_train_backward, y_train, X_test_backward, y_test, result_backward, p_values_backward, backward_selected_indices]

def train_final_model(lst, feature_names):
    X_train, y_train, X_test, y_test, result, p_values, selected_indices = lst
    y_pred = result.predict(sm.add_constant(X_test))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # 输出变量和对应的参数
    print('\n\n最终的模型信息')
    print(f"均方误差 (MSE): {mse}")
    print(f"决定系数 (R^2): {r2}")
    # 获取特征名称
    feature_names_with_const = ['const'] + [feature_names[i] for i in selected_indices]
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

# 训练最终模型 - 向前选择
forward_result = forward_selection_model(X_train_final.values, y_train, X_test_final.values, y_test, X_train_final.columns)
print("\n向前选择最终模型的系数和 p 值:")
train_final_model(forward_result, X_train_final.columns)

# # 训练最终模型 - 向后选择
# backward_result = backward_selection_model(X_train_final.values, y_train, X_test_final.values, y_test, X_train_final.columns)
# print("\n向后选择最终模型的系数和 p 值:")
# train_final_model(backward_result, X_train_final.columns)