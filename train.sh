DATA_ROOT=./dataset/tw
VAL_DATA_ROOT=./dataset/labeled/
LABEL_FILE=./dataset/labeled/labels.json
TRAIN_BATCH=80
VAL_BATCH=80
SEED=89112
N_EPOCHS=100
N_EPOCHS_DECAY=100
LR=0.0002
SAVE_ROOT=./LPDGAN/checkpoints
SAVE_EPOCH=2
TRAIN_DATASET=$DATA_ROOT/$1
GPUID=0
SAVE=$SAVE_ROOT/$2
PLATE_LOSS=l1
WARM_UP=1
PRETRAINED=./LPDGAN/checkpoints/pretrained
WBATCH=160

python train_LPDGAN.py \
    --data_root $TRAIN_DATASET \
    --txt_loss $PLATE_LOSS \
    --val_data_root $VAL_DATA_ROOT  --label_file $LABEL_FILE\
    --gpu_id $GPUID --batch_size $TRAIN_BATCH --val_batch $VAL_BATCH\
    --n_epochs $N_EPOCHS --n_epochs_decay $N_EPOCHS_DECAY \
    --pretrained_weights $PRETRAINED \
    --lr $LR --model_save_root $SAVE\
    --save_epoch_freq $SAVE_EPOCH\
    --seed $SEED --preload \
    --D_warm_up $WARM_UP --warm_up_batch_size $WBATCH 
