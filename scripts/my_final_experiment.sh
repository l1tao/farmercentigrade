# scripts/my_final_experiment.sh

# --- 1. 关键配置：请根据您生成的 .npy 文件形状来修改 ---

# a. 数据路径和名称
root_path=./dataset/preprocessed_data/   # .npy文件所在的文件夹
data=my_preprocessed                    # 我们在 data_factory.py 中注册的名字
data_path=placeholder.txt               # 该参数不会被使用，可以是任意字符串
# b. 序列长度和特征维度 (!! 必须与 final_X.npy 的形状 (样本数, 序列长度, 特征数) 匹配 !!)
seq_len=60    # 您的序列长度 (来自 SEQUENCE_LENGTH)
enc_in=20    # 您的特征数量 (X的最后一维)
dec_in=20     # 通常与 enc_in 相同

# c. 预测长度和目标维度 (!! 必须与 final_y.npy 的形状 (样本数, 预测长度, 目标数) 匹配 !!)
pred_len=1    # 您的预测长度 (来自 PREDICTION_LENGTH)
c_out=1       # 您预测的目标数量 (Y的最后一维，如果是(N, 1)则为1)

# d. 为本次训练运行创建一个唯一的ID
model_id=my_model_seq${seq_len}_pred${pred_len}

# --- 2. 模型超参数 (可以先保持默认) ---
model_name=TimeMixerPP
e_layers=2
d_model=16
d_ff=64
batch_size=32 # 如果内存不足可以调小


# --- 3. 执行命令 (通过 is_training 控制模式) ---
python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path $root_path \
  --data_path $data_path \
  --model_id $model_id \
  --model $model_name \
  --data $data \
  --features 'MS' \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --des 'FinalTrain' \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --down_sampling_layers 2 \
  --down_sampling_method 'avg' \
  --down_sampling_window 2


