export CUDA_VISIBLE_DEVICES=0

#cd ..

for pred_len in 24 12 20 30
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
  --data_name 'illness' \
  --seq_len 36 \
  --label_len 24 \
  --pred_len $pred_len \
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
  --patch_len 16\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 2\
  --info 'TimeMixer flexiable_prediction_1 for ILI 10.5'\

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
  --seq_len 36 \
  --label_len 24 \
  --pred_len $pred_len \
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
  --patch_len 16\
  --stride 8\
  --fourier_decomp_ratio 0.5\
  --n_heads 3\
  --T2V_outmodel 12\
  --info 'FITS flexiable_prediction_1 for ILI 10.5'\

done


done



