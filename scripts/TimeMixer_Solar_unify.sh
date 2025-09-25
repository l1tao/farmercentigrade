model_name=TimeMixerPP
seq_len=96
down_sampling_layers=2
down_sampling_window=2
e_layers=2
d_model=16
d_ff=64


root_path=./dataset/solar/
data_path=solar_AL.txt

python -u run.py \
  --task_name long_term_forecast \
  --root_path $root_path \
  --data_path $data_path \
  --model_id solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --use_norm 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --channel_independence 1 \
  --channel_mixing 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --root_path $root_path \
  --data_path $data_path \
  --model_id solar_96_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 192 \
  --e_layers $e_layers \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --use_norm 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --channel_independence 1 \
  --channel_mixing 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --root_path $root_path \
  --data_path $data_path \
  --model_id solar_96_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 336 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --use_norm 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --channel_independence 1 \
  --channel_mixing 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window

python -u run.py \
  --task_name long_term_forecast \
  --root_path $root_path \
  --data_path $data_path \
  --model_id solar_96_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --use_norm 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --channel_independence 1 \
  --channel_mixing 1 \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method conv \
  --down_sampling_window $down_sampling_window
