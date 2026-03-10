export CUDA_VISIBLE_DEVICES=0

#cd ..

for pred_len in 24 20 30 12
do

for rate in    0.01 0.001 0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'DeepTD_patch_koopa_V2'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --loss quantile \
  --seed 1 \
  --data_name 'illness' \
  --seq_len 36 \
  --label_len 12 \
  --pred_len $pred_len \
  --d2v_train_pred_len 24 \
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 6\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 2\
  --T2V_outmodel 12\
  --info 'D2Vformer flexiable_prediction_1 for illness'\

done

for rate in     0.0001 0.00005 0.00001 0.000005 0.000001
do

python -u main.py  \
  --model_name 'Autoformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --loss quantile \
  --seed 1 \
  --data_name 'illness' \
  --seq_len 36 \
  --label_len 12 \
  --pred_len $pred_len \
  --d2v_train_pred_len 24 \
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 6\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel 12\
  --info 'Autoformer flexiable_prediction_1 for illness'\

done

for rate in     0.0001 0.00005 0.00001 0.000005 0.000001
do

python -u main.py  \
  --model_name 'Fedformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --loss quantile \
  --seed 1 \
  --data_name 'illness' \
  --seq_len 36 \
  --label_len 12 \
  --pred_len $pred_len \
  --d2v_train_pred_len 24 \
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 6\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel 12\
  --info 'Fedformer flexiable_prediction_1 for illness'\

done

for rate in     0.0001 0.00005 0.00001 0.000005 0.000001
do

python -u main.py  \
  --model_name 'DLinear'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --loss quantile \
  --seed 1 \
  --data_name 'illness' \
  --seq_len 36 \
  --label_len 12 \
  --pred_len $pred_len \
  --d2v_train_pred_len 24 \
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 6\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 2\
  --T2V_outmodel 12\
  --info 'DLinear flexiable_prediction_1 for illness'\

done

for rate in     0.0001 0.00005 0.00001 0.000005 0.000001
do

python -u main.py  \
  --model_name 'PatchTST'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --loss quantile \
  --seed 1 \
  --data_name 'illness' \
  --seq_len 36 \
  --label_len 12 \
  --pred_len $pred_len \
  --d2v_train_pred_len 24 \
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 6\
  --stride 2\
  --fourier_decomp_ratio 0.5\
  --n_heads 2\
  --T2V_outmodel 12\
  --info 'PatchTST flexiable_prediction_1 for illness'\

done


done
