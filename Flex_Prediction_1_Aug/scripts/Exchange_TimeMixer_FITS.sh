export CUDA_VISIBLE_DEVICES=0

#cd ..

for pred_len in 96 72 108 48
do

for rate in     0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do
python -u main.py \
  --model_name 'TimeMixer'\
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
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 16\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 2\
  --info 'TimeMixer flexiable_prediction_1 for exchange 10.5'\

done

for rate in     0.0005  0.0001 0.00005 0.00001 0.000005 0.000001
do

python -u main.py  \
  --model_name 'FITS'\
  --exp flexiable_prediction_1\
  --train True \
  --resume False \
  --loss quantile \
  --seed 1 \
  --data_name 'exchange' \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --d2v_train_pred_len 96 \
  --d_feature 8 \
  --c_out 8 \
  --features 'M' \
  --d_model 512 \
  --d_ff 2048 \
  --lr $rate \
  --batch_size 64 \
  --patience 5 \
  --e_layers 2 \
  --d_layers 1 \
  --dropout 0.1\
  --patch_len 16\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel 12\
  --info 'FITS flexiable_prediction_1 for exchange 10.5'\

done


done



