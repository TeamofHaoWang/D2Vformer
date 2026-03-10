export CUDA_VISIBLE_DEVICES=0

#cd ..

for n_T2V in 64 128 256
do
for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py  \
  --model_name 'D2Vformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'illness' \
  --seq_len 36 \
  --pred_len 24,32,40,44,46,48,50,52,56,64,72 \
  --d2v_train_pred_len 48\
  --d_feature 7 \
  --c_out 7 \
  --features 'M' \
  --d_model 128 \
  --d_ff 256 \
  --lr $rate \
  --batch_size 32 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 4\
  --stride 2\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --down_sampling_layers 2 \
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
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 6,8,10,12,14,16,18\
  --d2v_train_pred_len 12\
  --d_feature 8 \
  --c_out 8 \
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
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 8,12,14,16,18,20,24\
  --d2v_train_pred_len 16\
  --d_feature 8 \
  --c_out 8 \
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
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 12,18,24,30,36\
  --d2v_train_pred_len 24\
  --d_feature 8 \
  --c_out 8 \
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
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 24,32,40,44,48,52,56,64,72\
  --d2v_train_pred_len 48\
  --d_feature 8 \
  --c_out 8 \
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
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 48,56,64,72,80,88,92,96,100,104,112,120,128,136,144\
  --d2v_train_pred_len 96\
  --d_feature 8 \
  --c_out 8 \
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
  --augmentation True\
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --pred_len 64,76,88,100,112,144,156,168,180,192\
  --d2v_train_pred_len 128\
  --d_feature 8 \
  --c_out 8 \
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
  --info 'D2Vformer augmentation flexiable_prediction_1 8.19'\

done
done






#for n_T2V in 64 128 256
#do
#for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
#do
#python -u main.py  \
#  --model_name 'D2Vformer'\
#  --exp flexiable_prediction_1\
#  --train True \
#  --resume False \
#  --augmentation True\
#  --loss quantile \
#  --seed 1 \
#  --data_name 'illness' \
#  --seq_len 36 \
#  --pred_len 4,6,8,10,12 \
#  --d2v_train_pred_len 8\
#  --d_feature 7 \
#  --c_out 7 \
#  --features 'M' \
#  --d_model 128 \
#  --d_ff 256 \
#  --lr $rate \
#  --batch_size 64 \
#  --patience 5 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --dropout 0.1\
#  --patch_len 4\
#  --stride 2\
#  --fourier_decomp_ratio 0.5\
#  --n_heads 3\
#  --T2V_outmodel $n_T2V\
#  --down_sampling_layers 2 \
#  --info 'D2Vformer augmentation flexiable_prediction_1 8.18'\
#
#done
#done
#
#for n_T2V in 64 128 256
#do
#for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
#do
#python -u main.py  \
#  --model_name 'D2Vformer'\
#  --exp flexiable_prediction_1\
#  --train True \
#  --resume False \
#  --augmentation True\
#  --loss quantile \
#  --seed 1 \
#  --data_name 'illness' \
#  --seq_len 36 \
#  --pred_len 6,9,12,15,18 \
#  --d2v_train_pred_len 12\
#  --d_feature 7 \
#  --c_out 7 \
#  --features 'M' \
#  --d_model 128 \
#  --d_ff 256 \
#  --lr $rate \
#  --batch_size 64 \
#  --patience 5 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --dropout 0.1\
#  --patch_len 4\
#  --stride 2\
#  --fourier_decomp_ratio 0.5\
#  --n_heads 3\
#  --T2V_outmodel $n_T2V\
#  --down_sampling_layers 2 \
#  --info 'D2Vformer augmentation flexiable_prediction_1 8.18'\
#
#done
#done
#
#
#for n_T2V in 64 128 256
#do
#for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
#do
#python -u main.py  \
#  --model_name 'D2Vformer'\
#  --exp flexiable_prediction_1\
#  --train True \
#  --resume False \
#  --augmentation True\
#  --loss quantile \
#  --seed 1 \
#  --data_name 'illness' \
#  --seq_len 36 \
#  --pred_len 9,12,14,15,16,18,20,21,22,24,27 \
#  --d2v_train_pred_len 18\
#  --d_feature 7 \
#  --c_out 7 \
#  --features 'M' \
#  --d_model 128 \
#  --d_ff 256 \
#  --lr $rate \
#  --batch_size 64 \
#  --patience 5 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --dropout 0.1\
#  --patch_len 4\
#  --stride 2\
#  --fourier_decomp_ratio 0.5\
#  --n_heads 3\
#  --T2V_outmodel $n_T2V\
#  --down_sampling_layers 2 \
#  --info 'D2Vformer augmentation flexiable_prediction_1 8.18'\
#
#done
#done
#
#
#for n_T2V in 64 128 256
#do
#for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
#do
#python -u main.py  \
#  --model_name 'D2Vformer'\
#  --exp flexiable_prediction_1\
#  --train True \
#  --resume False \
#  --augmentation True\
#  --loss quantile \
#  --seed 1 \
#  --data_name 'illness' \
#  --seq_len 36 \
#  --pred_len 10,12,14,16,18,20,22,24,26,28,30 \
#  --d2v_train_pred_len 20\
#  --d_feature 7 \
#  --c_out 7 \
#  --features 'M' \
#  --d_model 128 \
#  --d_ff 256 \
#  --lr $rate \
#  --batch_size 64 \
#  --patience 5 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --dropout 0.1\
#  --patch_len 4\
#  --stride 2\
#  --fourier_decomp_ratio 0.5\
#  --n_heads 3\
#  --T2V_outmodel $n_T2V\
#  --down_sampling_layers 2 \
#  --info 'D2Vformer augmentation flexiable_prediction_1 8.18'\
#
#done
#done
#
#
#for n_T2V in 64 128 256
#do
#for rate in  0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
#do
#python -u main.py  \
#  --model_name 'D2Vformer'\
#  --exp flexiable_prediction_1\
#  --train True \
#  --resume False \
#  --augmentation True\
#  --loss quantile \
#  --seed 1 \
#  --data_name 'illness' \
#  --seq_len 36 \
#  --pred_len 18,27,28,32,36,40,44,45,54 \
#  --d2v_train_pred_len 36\
#  --d_feature 7 \
#  --c_out 7 \
#  --features 'M' \
#  --d_model 128 \
#  --d_ff 256 \
#  --lr $rate \
#  --batch_size 64 \
#  --patience 5 \
#  --e_layers 2 \
#  --d_layers 1 \
#  --dropout 0.1\
#  --patch_len 4\
#  --stride 2\
#  --fourier_decomp_ratio 0.5\
#  --n_heads 3\
#  --T2V_outmodel $n_T2V\
#  --down_sampling_layers 2 \
#  --info 'D2Vformer augmentation flexiable_prediction_1 8.18'\
#
#done
#done




