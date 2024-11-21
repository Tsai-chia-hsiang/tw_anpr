DATA_ROOT=./dataset/labeled/
LABEL=./dataset/labeled/北鎮所_labels.json
DB_CKPT=$1
GPUID=$2
python ocr_eval.py --data_root $DATA_ROOT --label_file $LABEL\
     --deblur_weight_dir $DB_CKPT --gpu $GPUID