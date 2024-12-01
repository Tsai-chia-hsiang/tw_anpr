DATA_ROOT=./dataset/tw
TXT_EXT=paddleocr
TXT_CACHED=$DATA_ROOT/$1/paddleocr.pth
VAL_DATA_ROOT=./dataset/labeled/
LABEL_FILE=./dataset/labeled/北鎮所_labels.json
TRAIN_BATCH=196
VAL_BATCH=196
SEED=89112
N_EPOCHS=20
N_EPOCHS_DECAY=40
LR=0.0001
SAVE_ROOT=./LPDGAN/checkpoints
SAVE_EPOCH=2
TRAIN_DATASET=$DATA_ROOT/$1
GPUID=cuda
SAVE=$SAVE_ROOT/$2
PLATE_LOSS=probl1
WARM_UP=2
PRETRAINED=$SAVE_ROOT/pretrained
WBATCH=320
    
CUDA_VISIBLE_DEVICES=0,1,3,4 python train_LPDGAN.py \
    --data_root $TRAIN_DATASET \
    --txt_cached $TXT_CACHED --txt_extract $TXT_EXT --txt_loss $PLATE_LOSS \
    --val_data_root $VAL_DATA_ROOT  --label_file $LABEL_FILE\
    --gpu_id $GPUID --batch_size $TRAIN_BATCH --val_batch $VAL_BATCH\
    --n_epochs $N_EPOCHS --n_epochs_decay $N_EPOCHS_DECAY \
    --pretrained_weights $PRETRAINED --load_G \
    --lr $LR --model_save_root $SAVE\
    --save_epoch_freq $SAVE_EPOCH\
    --seed $SEED --preload \
    --D_warm_up $WARM_UP --warm_up_batch_size $WBATCH 
