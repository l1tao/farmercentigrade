# merge_files.py (最新版：删除高NaN比例列 + 删除时间列)
import pandas as pd
import glob
import os

# --- 配置 ---
# 存放所有xlsx文件的文件夹路径
excel_files_path = 'dataset/One Minute Data/*.xlsx'  # <--- 修改为您存放30个xlsx文件的文件夹路径

# 合并后输出的CSV文件名
output_csv_file = 'dataset/merged_cleaned_data.csv'   # 清洗后输出的CSV文件名

# !! 关键配置 !!
TIME_COL_NAME = 'time_UTC_OFS__0400_'         # 指定要删除的时间列的名称
NAN_THRESHOLD = 0.5            # 删除标准：如果一列中NaN的比例超过50% (0.5)，就删除它

# --- 主流程 ---
all_files = glob.glob(excel_files_path)
if not all_files:
    print(f"错误：在路径 '{excel_files_path}' 中没有找到任何 .xlsx 文件。请检查路径。")
else:
    print(f"找到 {len(all_files)} 个xlsx文件。开始合并...")
    
    # --- 步骤1：找出所有文件中出现过的全部列名 ---
    all_columns = set()
    for filepath in all_files:
        try:
            df_cols = pd.read_excel(filepath, nrows=0).columns
            all_columns.update(df_cols)
        except Exception as e:
            print(f"读取文件 {filepath} 列名时出错: {e}")
            continue
    
    master_column_list = sorted(list(all_columns))
    print(f"已识别出所有文件中的唯一列名集合，共 {len(master_column_list)} 列。")

    df_list = []
    # --- 步骤2：循环读取文件，并统一列结构 ---
    for filepath in all_files:
        filename = os.path.basename(filepath)
        try:
            person_id = int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            person_id = filename
        
        try:
            df = pd.read_excel(filepath)
            df = df.reindex(columns=master_column_list)
            df['person_id'] = person_id
            df_list.append(df)
            print(f"  已处理文件: {filename}, ID: {person_id}")
        except Exception as e:
            print(f"处理文件 {filepath} 时出错: {e}")
            continue

    if df_list:
        merged_df = pd.concat(df_list, ignore_index=True)
        print(f"\n文件合并完成。合并后形状: {merged_df.shape}")

        # --- 步骤3：删除NaN比例过高的列 ---
        print(f"\n开始检查缺失值超过 {NAN_THRESHOLD:.0%} 的列...")
        nan_percentages = merged_df.isnull().sum() / len(merged_df)
        cols_to_drop_nan = nan_percentages[nan_percentages > NAN_THRESHOLD].index.tolist()
        
        if cols_to_drop_nan:
            print(f"将删除以下缺失率过高的列: {cols_to_drop_nan}")
            merged_df = merged_df.drop(columns=cols_to_drop_nan)
        else:
            print("没有发现缺失率过高的列。")

        # --- 步骤4：删除时间列 ---
        if TIME_COL_NAME in merged_df.columns:
            print(f"正在删除时间列: '{TIME_COL_NAME}'")
            merged_df = merged_df.drop(columns=[TIME_COL_NAME])
        else:
            print(f"警告：数据中未找到名为 '{TIME_COL_NAME}' 的时间列，无法删除。")
            
        # --- 步骤5：整理并保存 ---
        if 'person_id' in merged_df.columns:
            cols = ['person_id'] + [col for col in merged_df if col != 'person_id']
            merged_df = merged_df[cols]
        
        merged_df.to_csv(output_csv_file, index=False)

        print(f"\n清洗完成！最终数据已保存到: {output_csv_file}")
        print(f"清洗后的数据形状: {merged_df.shape}")
    else:
        print("没有成功处理任何文件，无法合并。")