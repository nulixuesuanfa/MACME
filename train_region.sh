DATASET_NAME='f30k'
DATA_PATH='/media/hdd4/luz/data/data/'${DATASET_NAME}
VOCAB_PATH='/media/hdd4/luz/data/vocab/'
gpu=0

#lamda1=(0.005)
#lamda2=(0.001)
#lamda3=(0.01)
#for((i=0;i<=0;i++))
#do
#CUDA_VISIBLE_DEVICES=$gpu python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
#  --logger_name runs/${DATASET_NAME}_butd_region_bigru_${lamda1[i]}_${lamda2[i]}_${lamda3[i]}/log --model_name runs/${DATASET_NAME}_butd_region_bigru_${lamda1[i]}_${lamda2[i]}_${lamda3[i]} \
#  --num_epochs=30 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 0 \
#  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1  --lamda1 ${lamda1[i]} --lamda2 ${lamda2[i]}  --lamda3 ${lamda3[i]}
#CUDA_VISIBLE_DEVICES=$gpu python eval.py --data_path $DATA_PATH --dataset $DATASET_NAME --model_path runs/${DATASET_NAME}_butd_region_bigru_${lamda1[i]}_${lamda2[i]}_${lamda3[i]}
#done

#lamda1=(0.5 0.5 0.5)
#lamda2=(0.1 0.01 0.5)
#lamda3=(0.5 0.001 0.1)
#for((i=0;i<=2;i++))
#do
#CUDA_VISIBLE_DEVICES=$gpu python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
#  --logger_name runs/${DATASET_NAME}_butd_region_bigru_${lamda1[i]}_${lamda2[i]}_${lamda3[i]}/log --model_name runs/${DATASET_NAME}_butd_region_bigru_${lamda1[i]}_${lamda2[i]}_${lamda3[i]} \
#  --num_epochs=30 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 2 \
#  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1  --lamda1 ${lamda1[i]} --lamda2 ${lamda2[i]}  --lamda3 ${lamda3[i]}
#CUDA_VISIBLE_DEVICES=$gpu python eval.py --data_path $DATA_PATH --dataset $DATASET_NAME --model_path runs/${DATASET_NAME}_butd_region_bigru_${lamda1[i]}_${lamda2[i]}_${lamda3[i]}
#done

for lamda1 in 0.005
do
for lamda2 in 0.003
do
for lamda3 in 0.01
do
#CUDA_VISIBLE_DEVICES=$gpu python3 train.py \
#  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
#  --logger_name runs/${DATASET_NAME}_butd_region_bigru_${lamda1}_${lamda2}_${lamda3}/log --model_name runs/${DATASET_NAME}_butd_region_bigru_${lamda1}_${lamda2}_${lamda3} \
#  --num_epochs=30 --lr_update=15 --learning_rate=.0005 --precomp_enc_type basic --workers 2 \
#  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 1  --lamda1 $lamda1 --lamda2 $lamda2  --lamda3 $lamda3
CUDA_VISIBLE_DEVICES=$gpu python eval.py --data_path $DATA_PATH --dataset $DATASET_NAME --model_path runs/${DATASET_NAME}_butd_region_bigru_${lamda1}_${lamda2}_${lamda3}
done
done
done