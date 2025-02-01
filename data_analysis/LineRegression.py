from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_excel('../all_data.xlsx')

X = df[['colorfulness', 'brightness', 'quality', 'contrast','similarity',
        'timelength', 'topic_complexity', 'emoji_num', 'likes']]
y = df['witchnum']
y = y / 1000

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 生成交叉项
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
feature_names = poly.get_feature_names_out(X.columns)

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
    return mse, r2, result

# 向前选择变量
def forward_selection(X_train, y_train, X_test, y_test, feature_names):
    selected_features = []
    remaining_features = list(range(X_train.shape[1]))
    best_mse = float('inf')
    while remaining_features:
        best_feature = None
        for feature in remaining_features:
            temp_features = selected_features + [feature]
            X_train_subset = X_train[:, temp_features]
            X_test_subset = X_test[:, temp_features]
            mse, _, _ = evaluate_negative_binomial_model(X_train_subset, y_train, X_test_subset, y_test)
            if mse < best_mse:
                best_mse = mse
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
    best_mse, _, _ = evaluate_negative_binomial_model(X_train, y_train, X_test, y_test)
    while len(selected_features) > 1:
        worst_feature = None
        for feature in selected_features:
            temp_features = selected_features.copy()
            temp_features.remove(feature)
            X_train_subset = X_train[:, temp_features]
            X_test_subset = X_test[:, temp_features]
            mse, _, _ = evaluate_negative_binomial_model(X_train_subset, y_train, X_test_subset, y_test)
            if mse <= best_mse:
                best_mse = mse
                worst_feature = feature
        if worst_feature is not None:
            selected_features.remove(worst_feature)
        else:
            break
    return [feature_names[i] for i in selected_features]

def forward_selection_model():
    # 执行向前选择
    forward_selected_features = forward_selection(X_train_poly, y_train, X_test_poly, y_test, feature_names)
    print("向前选择的特征:", forward_selected_features)
    # 使用向前选择的特征重新训练模型
    forward_selected_indices = [list(feature_names).index(f) for f in forward_selected_features]
    X_train_forward = X_train_poly[:, forward_selected_indices]
    X_test_forward = X_test_poly[:, forward_selected_indices]
    mse_forward, r2_forward, result_forward = evaluate_negative_binomial_model(X_train_forward, y_train, X_test_forward, y_test)
    print(f"向前选择后模型的均方误差 (MSE): {mse_forward}")
    print(f"向前选择后模型的决定系数 (R^2): {r2_forward}")
    return [X_train_forward, y_train, X_test_forward, y_test, result_forward]

def backward_selection_model():
    # 执行向后选择
    backward_selected_features = backward_selection(X_train_poly, y_train, X_test_poly, y_test, feature_names)
    print("向后选择的特征:", backward_selected_features)
    # 使用向后选择的特征重新训练模型
    backward_selected_indices = [list(feature_names).index(f) for f in backward_selected_features]
    X_train_backward = X_train_poly[:, backward_selected_indices]
    X_test_backward = X_test_poly[:, backward_selected_indices]
    mse_backward, r2_backward, result_backward = evaluate_negative_binomial_model(X_train_backward, y_train, X_test_backward, y_test)
    print(f"向后选择后模型的均方误差 (MSE): {mse_backward}")
    print(f"向后选择后模型的决定系数 (R^2): {r2_backward}")

    # 获取最终选择特征的 p 值
    feature_names_with_const = ['const'] + backward_selected_features
    p_values = result_backward.pvalues
    print("\n最终选择的特征及其 p 值：")
    for feature, p_value in zip(feature_names_with_const, p_values):
        print(f'{feature}: {p_value}')

    return [X_train_backward, y_train, X_test_backward, y_test, result_backward]

def train_final_model(lst):
    X_train, y_train, X_test, y_test, result = lst
    y_pred = result.predict(sm.add_constant(X_test))
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # 输出变量和对应的参数
    print('\n\n最终的模型信息')
    print(f"均方误差 (MSE): {mse}")
    print(f"决定系数 (R^2): {r2}")
    # 获取特征名称
    feature_names_with_const = ['const'] + list(poly.get_feature_names_out(X.columns))
    # 获取模型系数
    coefficients = result.params
    # 打印每个自变量对应的参数值
    for feature, coef in zip(feature_names_with_const, coefficients):
        print(f'{feature}: {coef}')

forward_result = forward_selection_model()
# backward_result = backward_selection_model()