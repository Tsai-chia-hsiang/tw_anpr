from anpr import LicensePlate_OCR, license_plate_ocr_evaluation
from LPDGAN.LPDGAN import SwinTrans_G, LPDGAN_DEFALUT_CKPT_DIR
from imgproc_utils import L_CLAHE, normalize_brightness
from pathlib import Path
import cv2
import json
from tqdm import tqdm
import argparse
from argparse import Namespace

def read_json(file:str):
    with open(file, 'r') as fp:
        return json.load(fp)

def main(args:Namespace):
    
    img_label:dict[str, str] = read_json(args.label_file)
    
    imgs = list(map(lambda x:args.data_root/x, list(img_label.keys())))
    labels = list(img_label.values())
    lp_ocr = LicensePlate_OCR()
    deblur_swintransformer = None
    if args.deblur is not None:
        deblur_swintransformer = SwinTrans_G(pretrained_ckpt=LPDGAN_DEFALUT_CKPT_DIR/args.deblur, mode='inference')
    
    pred = [None] * len(imgs)
    for i, img_path in enumerate(tqdm(imgs)):
        src_img = cv2.imread(img_path)
        if deblur_swintransformer is not None:
            src_img = deblur_swintransformer.inference(x=src_img)
        else:
            if src_img.shape[:2] != (112, 224):
                src_img = cv2.resize(src_img, (224,112))
            src_img = normalize_brightness(src_img)
        
        src_img = L_CLAHE(src_img)
        plate_number = lp_ocr(crop=src_img)
        pred[i] = plate_number[0] if plate_number[0] != 'n' else ''
    acc, all_correct, _ = license_plate_ocr_evaluation(pred=pred, gth=labels)

    print(f"Accuracy : {acc}, all correction rate: {all_correct}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--label_file", type=Path)
    parser.add_argument("--deblur", type=Path, default=None)
    args = parser.parse_args()
    main(args = args)