DATA_ROOT=./dataset/tw
VAL_DATA_ROOT=./dataset/labeled/
LABEL_FILE=./dataset/labeled/北鎮所_labels.json
TRAIN_BATCH=60
VAL_BATCH=60
N_EPOCHS=0
PRETRAIN=./LPDGAN/checkpoints/old_dataset/latest
CKPT=./LPDGAN/checkpoints/old_dataset/latest
N_EPOCHS_DECAY=200
LR=0.0002
SAVE_ROOT=./LPDGAN/checkpoints
SAVE_FEQ=5000
SAVE_EPOCH=1
TRAIN_DATASET=$DATA_ROOT/$1
GPUID=$2
SAVE=$SAVE_ROOT/$3

python train_LPDGAN.py --data_root $TRAIN_DATASET \
    --val_data_root $VAL_DATA_ROOT  --label_file $LABEL_FILE\
    --gpu_id $GPUID --batch_size $TRAIN_BATCH --val_batch $VAL_BATCH\
    --n_epochs $N_EPOCHS --n_epochs_decay $N_EPOCHS_DECAY \
    --lr $LR --model_save_root $SAVE\
    --pretrained_weights $PRETRAIN --checkpoint_dir $CKPT \
    --save_iter_freq $SAVE_FEQ --save_epoch_freq $SAVE_EPOCH
