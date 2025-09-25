import pandas as pd
from sklearn.experimental import enable_iterative_imputer 
import numpy as np
from sklearn.impute import IterativeImputer
# from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge


# --- 1. 参数配置 ---
INPUT_FILE = 'dataset/merged_all_person_data_no_time.csv'  # <--- 修改为您的CSV文件路径
PERSON_ID_COL = 'person_id'            # <--- 您的用户ID列名
TIME_COL = 'time_UTC_OFS__0400_'                      # <--- 您的时间列名
TARGET_COL = 'CorrBodyTemp(C)'                # <--- 您要预测的目标列名
EXCLUDE_COLS = []                      # 其他需要排除的列

# 滑动窗口配置
SEQUENCE_LENGTH = 60  # 使用过去60分钟的数据
PREDICTION_LENGTH = 1   # 预测未来1分钟的数据（或其他值）

# 缺失值填充配置
# IMP_KERNEL = KNeighborsRegressor(n_neighbors=3)
IMP_KERNEL = BayesianRidge()
MAX_ITER = 100

# --- 2. 滑动窗口函数 ---
def create_sequences(data, seq_len, pred_len, target_col_index):
    """为单个用户的数据创建滑动窗口样本"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        # 输入特征：从i到i+seq_len
        X.append(data[i:(i + seq_len)])
        # 目标值：从i+seq_len到i+seq_len+pred_len的目标列
        y.append(data[(i + seq_len):(i + seq_len + pred_len), target_col_index])
    return np.array(X), np.array(y)

# --- 3. 主处理流程 ---

# 加载数据
print(f"成功加载文件: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, sep=',')

# 初始化用于存放所有用户样本的列表
all_X = []
all_y = []

# 按用户ID进行分组处理
grouped = df.groupby(PERSON_ID_COL)
print(f"发现 {len(grouped)} 个独立用户/序列。开始分别处理...")

for person_id, person_df in grouped:
    print(f"  正在处理用户: {person_id}")
    
    # a. 准备当前用户的数据 (移除ID和时间列)
    # 首先确定目标列的索引位置
    feature_df = person_df.drop(columns=[PERSON_ID_COL, TIME_COL] + EXCLUDE_COLS)
    target_col_index = feature_df.columns.get_loc(TARGET_COL)
    
    # b. 对当前用户的数据进行缺失值填充
    if feature_df.isnull().sum().sum() > 0:
        imp = IterativeImputer(max_iter=MAX_ITER, estimator=IMP_KERNEL)
        imputed_data = imp.fit_transform(feature_df.values)
    else:
        imputed_data = feature_df.values
        
    # c. 为当前用户的数据创建滑动窗口样本
    X_person, y_person = create_sequences(imputed_data, SEQUENCE_LENGTH, PREDICTION_LENGTH, target_col_index)
    
    # d. 将当前用户的样本添加到总列表中
    if len(X_person) > 0:
        all_X.append(X_person)
        all_y.append(y_person)

# 4. 合并所有用户的样本
final_X = np.concatenate(all_X, axis=0)
final_y = np.concatenate(all_y, axis=0)

print("\n数据预处理完成！")
print(f"最终特征 X 的形状: {final_X.shape}")
print(f"最终目标 y 的形状: {final_y.shape}")

# 示例解读: 假设每个用户有1000个时间点, 10个特征
# 每个用户会生成 1000 - 60 - 1 + 1 = 940 个样本
# 30个用户总共会生成 30 * 940 = 28200 个样本
# final_X.shape -> (28200, 60, 10)  (样本数, 序列长度, 特征数)
# final_y.shape -> (28200, 1)       (样本数, 预测长度)
# 4. 合并所有用户的样本
final_X = np.concatenate(all_X, axis=0)
final_y = np.concatenate(all_y, axis=0)

print("\n数据预处理完成！")
print(f"最终特征 X 的形状: {final_X.shape}")
print(f"最终目标 y 的形状: {final_y.shape}")

# --- 新增：将处理好的数组保存到文件 ---
output_dir = 'dataset/preprocessed_data'
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, 'final_X.npy'), final_X)
np.save(os.path.join(output_dir, 'final_y.npy'), final_y)
print(f"处理好的 X 和 y 已保存到 '{output_dir}' 文件夹中。")