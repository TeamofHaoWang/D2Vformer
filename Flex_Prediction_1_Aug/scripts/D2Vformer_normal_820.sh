export CUDA_VISIBLE_DEVICES=1

#cd ..
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'D2Vformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation False\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 18,20,22,24,26,28,30,32\
  --d2v_train_pred_len 12\
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer normal flexiable_prediction_1 8.19'\

done
done


for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'D2Vformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 10,12,14,16,18,20,22,24,26,28,30,32\
  --d2v_train_pred_len 12\
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer augmentation flexiable_prediction_1 8.19'\

done
done


for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'D2Vformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation False\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 40,44\
  --d2v_train_pred_len 24\
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer normal flexiable_prediction_1 8.19'\

done
done


for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'D2Vformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 12,18,24,30,36,40,44\
  --d2v_train_pred_len 24\
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer augmentation flexiable_prediction_1 8.19'\

done
done


for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'D2Vformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation False\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 76,80,84,88\
  --d2v_train_pred_len 48\
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer normal flexiable_prediction_1 8.19'\

done
done


for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'D2Vformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 40,44,48,52,56,64,72,76,80,84,88\
  --d2v_train_pred_len 48\
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer augmentation flexiable_prediction_1 8.19'\

done
done


















































