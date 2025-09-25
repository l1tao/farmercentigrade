# impu_new.py (最终稳健版)

import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer # <--- 使用更简单的 SimpleImputer

# --- 1. 参数配置 ---
# 输入文件：必须是上一步合并清洗后输出的文件
INPUT_FILE = 'dataset/merged_cleaned_data.csv'

# 输出文件夹：存放最终可用于训练的 X 和 y 数组
OUTPUT_DIR = 'dataset/preprocessed_data'

# 列名配置
PERSON_ID_COL = 'person_id'
TARGET_COL = 'CorrBodyTemp(C)'      # <--- !! 修改为您要预测的目标列名 !!

# 滑动窗口配置
SEQUENCE_LENGTH = 60
PREDICTION_LENGTH = 1

# --- 2. 滑动窗口函数 (无需修改) ---
def create_sequences(data, seq_len, pred_len, target_col_index):
    X, y = [], []
    if len(data) < seq_len + pred_len:
        return np.array([]), np.array([])

    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:(i + seq_len)])
        y.append(data[(i + seq_len):(i + seq_len + pred_len), target_col_index])
    return np.array(X), np.array(y)

# --- 3. 主处理流程 ---

print(f"加载文件: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, sep=',')

all_X, all_y = [], []
grouped = df.groupby(PERSON_ID_COL)
print(f"发现 {len(grouped)} 个独立用户/序列。开始分别处理...")

# 关键：在循环外创建一个插补器，并用所有数据进行训练
# 这样可以确保即使某个用户的数据很少，也能用全局的均值来填充
feature_cols_df = df.drop(columns=[PERSON_ID_COL])
imputer = SimpleImputer(strategy='mean')
print("正在使用全局数据训练均值插补器...")
imputer.fit(feature_cols_df)
print("插补器训练完成。")

for person_id, person_df in grouped:
    print(f"  正在处理用户: {person_id}")
    
    feature_df_person = person_df.drop(columns=[PERSON_ID_COL])
    
    if feature_df_person.empty:
        print(f"    用户 {person_id} 的数据为空，已跳过。")
        continue

    try:
        target_col_index = feature_df_person.columns.get_loc(TARGET_COL)
    except KeyError:
        print(f"错误：在数据列中找不到您指定的目标列 '{TARGET_COL}'。请检查配置。")
        exit()
    
    # 使用训练好的插补器对当前用户数据进行转换（填充）
    imputed_data = imputer.transform(feature_df_person)
        
    X_person, y_person = create_sequences(imputed_data, SEQUENCE_LENGTH, PREDICTION_LENGTH, target_col_index)
    
    if len(X_person) > 0:
        all_X.append(X_person)
        all_y.append(y_person)
        print(f"    为用户 {person_id} 创建了 {len(X_person)} 个训练样本。")
    else:
        print(f"    用户 {person_id} 的数据点不足以创建任何样本，已跳过。")

if all_X and all_y:
    final_X = np.concatenate(all_X, axis=0)
    final_y = np.concatenate(all_y, axis=0)

    print("\n数据预处理完成！")
    print(f"最终特征 X 的形状: {final_X.shape}")
    print(f"最终目标 y 的形状: {final_y.shape}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, 'final_X.npy'), final_X)
    np.save(os.path.join(OUTPUT_DIR, 'final_y.npy'), final_y)
    print(f"处理好的 X 和 y 已保存到 '{OUTPUT_DIR}' 文件夹中。")
else:
    print("\n处理结束，但未能生成任何有效的训练样本。请检查您的数据和参数设置。")