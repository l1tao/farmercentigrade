import pandas as pd

# --- 请将这里的路径修改为您文件的实际路径 ---
file_path = 'dataset/merged_1_minute_data_no_lag.csv'

try:
    # 读取您的CSV文件
    df = pd.read_csv(file_path)

    # 检查整个文件中缺失值的总数
    total_missing_values = df.isnull().sum().sum()

    if total_missing_values == 0:
        print("✅ 文件已经清洗干净，没有任何缺失值 (NaNs)。")
    else:
        print(f"❌ 文件尚未完全清洗，共发现 {total_missing_values} 个缺失值。")
        print("\n每列的缺失值数量如下：")
        # 打印出每一列具体有多少个缺失值
        print(df.isnull().sum())

except FileNotFoundError:
    print(f"错误：找不到文件 '{file_path}'。请确保路径正确。")
except Exception as e:
    print(f"读取文件时发生错误: {e}")