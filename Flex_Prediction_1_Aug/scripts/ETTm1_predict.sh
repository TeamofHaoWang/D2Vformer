export CUDA_VISIBLE_DEVICES=1

for model_name in D2Vformer DeepTD_non_blind DeepTD_non_blind_V2
do
for n_T2V in 64 128 256
do
for rate in  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 8,12,14,16,18,20,24,26\
  --d2v_train_pred_len 16\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer augmentation 8.22'\

done
done
done


for model_name in D2Vformer DeepTD_non_blind DeepTD_non_blind_V2
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 12,18,24,30,36,40\
  --d2v_train_pred_len 24\
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
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer augmentation 8.22'\

done
done
done



for model_name in D2Vformer DeepTD_non_blind DeepTD_non_blind_V2
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 24,32,40,44,48,52,56,64,72,76,80\
  --d2v_train_pred_len 48\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer augmentation 8.22'\

done
done
done


for model_name in D2Vformer DeepTD_non_blind DeepTD_non_blind_V2
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 48,56,64,72,80,88,92,96,100,104,112,120,128,136,144,152,156,160\
  --d2v_train_pred_len 96\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer augmentation 8.22'\

done
done
done




for model_name in D2Vformer DeepTD_non_blind DeepTD_non_blind_V2
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 64,76,88,100,112,120,124,128,132,136,144,156,168,180,192\
  --d2v_train_pred_len 128\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer augmentation 8.22'\

done
done
done








for model_name in D2Vformer
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation False\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 8,12,14,16,18,20,24,26\
  --d2v_train_pred_len 16\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer 8.22'\

done
done
done


for model_name in D2Vformer
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation False\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 12,18,24,30,36,40\
  --d2v_train_pred_len 24\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer 8.22'\

done
done
done



for model_name in D2Vformer
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation False\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 24,32,40,44,48,52,56,64,72,76,80\
  --d2v_train_pred_len 48\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer 8.22'\

done
done
done


for model_name in D2Vformer
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation False\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 48,56,64,72,80,88,92,96,100,104,112,120,128,136,144,152,156,160\
  --d2v_train_pred_len 96\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --batch_size 128 \
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
  --info 'D2Vformer 8.22'\

done
done
done




for model_name in D2Vformer
do
for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name $model_name\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation False\
  --loss quantile \
  --seed 1 \
  --data_name 'ETTm1' \
  --seq_len 96 \
  --pred_len 64,76,88,100,112,120,124,128,132,136,144,156,168,180,192\
  --d2v_train_pred_len 128\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 8\
  --stride 4\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer 8.22'\

done
done
done

















