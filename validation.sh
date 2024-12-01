DATA_ROOT=./dataset/labeled/
LABEL=./dataset/labeled/labels.json
#DB_CKPT=$1
#GPUID=$2
python ocr_eval.py --data_root $DATA_ROOT --label_file $LABEL --gpu 0 \
     --ckpt ./LPDGAN/checkpoints/paddle_ocr_current_best/lcs/net_G.pth
     # --ckpt ./LPDGAN/checkpoints/old_paddle_please_gogo/lcs/net_G.pth
     #--ckpt ./LPDGAN/checkpoints/easyocr_new_l1_overfitting/lcs/net_G.pth

     
