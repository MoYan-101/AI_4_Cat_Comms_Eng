import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

np.random.seed(42)

df = pd.read_csv("Main_v1.csv")

# 输出包含空值的列名


# 数值类型的列
number_col_names = []
type_col_names = []
cols_with_na = df.columns[df.isna().any()]
for col in cols_with_na:
    # print(col)
    if "ame" in col:
        continue
    if "Type" in col:
        type_col_names.append(col)
    else:
        number_col_names.append(col)

print("type col:", type_col_names, "number col:", number_col_names)

# exit()

# 处理数值类型数据
for col_name in number_col_names:
    col = df[col_name]

    observed = col.dropna().values

    # 1. 拟合KDE
    kde = gaussian_kde(observed)

    # 2. 蒙特卡洛采样
    n_missing = col.isna().sum()
    samples = kde.resample(n_missing).flatten()
    if 'Calcination time' in col_name:
        samples = np.clip(np.round(samples), 0, 200)
    else:
        samples = np.clip(samples, 0, 1e9)

    # 3. 填充空缺
    mask = col.isna()

    df.loc[mask, col_name] = samples

# 处理非数值类型数据
for col_name in type_col_names:
    col = df[col_name]
    non_null = col.dropna()

    # 统计类别概率
    vals, probs = np.unique(non_null, return_counts=True)
    probs = probs / probs.sum()
    # 随机采样
    n_missing = col.isna().sum()
    samples = np.random.choice(vals, size=n_missing, p=probs)
    # 填充
    df.loc[col.isna(), col_name] = samples


df.to_csv("data_filled.csv", index=False)

