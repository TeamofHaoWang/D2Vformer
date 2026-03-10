export CUDA_VISIBLE_DEVICES=1

#cd ..

for n_T2V in 64 128 256
do
for pred_len in 96 72 108 48
do
for rate in     0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'D2Vformer'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 16\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --info 'D2Vformer flexiable_prediction_1 for exchange 2.8'\

done
done
done


for n_T2V in 64 128 256
do
for pred_len in 12 24 20 30
do
for rate in     0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py  \
  --model_name 'D2Vformer'\
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
  --batch_size 128 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 16\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel $n_T2V\
  --down_sampling_layers 2 \
  --info 'D2Vformer flexiable_prediction_1 for ILI 2.8'\

done
done
done



