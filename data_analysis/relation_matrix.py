import pandas as pd
import plotly.express as px

# 读取数据
df = pd.read_excel('../all_data.xlsx')

# 划分自变量和因变量
X = df[['EC', 'OC', 'colorfulness', 'brightness', 'quality','contrast','similarity',
        'timelength', 'topic_complexity', 'emoji_num','likes']]
y = df['witchnum']


# 将 X 和 y 合并成一个新的 DataFrame
combined_df = pd.concat([X, y], axis=1)
# 重命名列名，这里假设新列名为 'witchnum_scaled'
combined_df.rename(columns={combined_df.columns[-1]: 'witchnum_scaled'}, inplace=True)

# 计算相关矩阵
correlation_matrix = combined_df.corr()

# 创建 Plotly 热力图，通过修改 zmin 和 zmax 让颜色变浅
fig = px.imshow(correlation_matrix,
                labels=dict(x="变量", y="变量", color="相关性"),
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-0.8,  # 调整颜色映射范围，让颜色变浅
                zmax=0.8)

# 添加数字标签
for i, row in enumerate(correlation_matrix.values):
    for j, value in enumerate(row):
        fig.add_annotation(
            x=correlation_matrix.columns[j],
            y=correlation_matrix.index[i],
            text=f'{value:.2f}',
            showarrow=False,
            font=dict(color='black' if abs(value) < 0.5 else 'white')  # 根据颜色深浅调整文本颜色
        )

# 设置标题并使其居中
fig.update_layout(
    title={
        'text': '相关矩阵图',
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

# 显示图形
fig.show()