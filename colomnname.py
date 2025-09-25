# 可以用这个小脚本快速查看列名
import pandas as pd
df = pd.read_csv('dataset/merged_all_person_data.csv')
print(df.columns)