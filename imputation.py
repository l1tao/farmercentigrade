# 1. 导入所有需要的包
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 2. 定义缺失的变量 (请根据你的实际情况修改这些值)
INPUT_FILE = 'dataset/merged_1_minute_data_no_lag.csv'  # <-- 替换成你的输入文件名
EXCLUDE_COL = ['col_to_exclude_1', 'col_to_exclude_2']  # <-- 替换成你想要移除的列名列表
THRESHOLD = -10.0  # <-- 替换成你用于过滤温度的阈值

# ------------------- 以下是您提供的原始代码 -------------------

# this is for imputation preprocessing
MAX_ITER = 100
# IMP_KERNEL = DecisionTreeRegressor(max_features='sqrt')
IMP_KERNEL = KNeighborsRegressor(n_neighbors=3)
# IMP_KERNEL = SVR(C=0.5, epsilon=0.25, gamma='scale')
# IMP_KERNEL = MLPRegressor(learning_rate='adaptive', max_iter=500)
# IMP_KERNEL = GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100)

# this is for prediction
FOLD_NUM = 5
# REG_KERNEL = SVR(kernel='poly', degree=3)
# REG_KERNEL = DecisionTreeRegressor(max_features='sqrt')
# REG_KERNEL = MLPRegressor(learning_rate='adaptive', max_iter=2000)
REG_KERNEL = GradientBoostingRegressor(n_estimators=400, loss='squared_error', learning_rate=0.2,
                                     n_iter_no_change=None)
# REG_KERNEL = KNeighborsRegressor(n_neighbors=3)

REG_KERNEL = make_pipeline(StandardScaler(), REG_KERNEL)

# 确保文件存在，否则会报错
try:
    df = pd.read_csv(INPUT_FILE, sep=',')
    print(f"成功加载文件: {INPUT_FILE}")
except FileNotFoundError:
    print(f"错误: 找不到文件 '{INPUT_FILE}'。请确保INPUT_FILE变量路径正确。")
    exit()


# remove not wanted cols
# 确保要排除的列存在于DataFrame中
existing_exclude_cols = [col for col in EXCLUDE_COL if col in df.columns]
df = df.drop(columns=existing_exclude_cols)

# # if the target is mPSI, run this line
# df = df.dropna(subset=['mPSI'])

# remove the time
df_no_time = df.iloc[:, 1:]

# impute missing values here
imp = IterativeImputer(max_iter=MAX_ITER, estimator=IMP_KERNEL)
print("开始填充缺失值...")
input_no_NAN = imp.fit_transform(df_no_time.values)
print("缺失值填充完成。")

# remove data point with too low temperature
print(f"过滤前数据形状: {input_no_NAN.shape}")
input_no_NAN = input_no_NAN[input_no_NAN[:, 0] >= THRESHOLD]
print(f"过滤后数据形状 (第一列 >= {THRESHOLD}): {input_no_NAN.shape}")

y = input_no_NAN[:, 0]
X = input_no_NAN[:, 1:]

print("\n数据预处理完成！")
print(f"特征 X 的形状: {X.shape}")
print(f"目标 y 的形状: {y.shape}")