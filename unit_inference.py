import numpy as np
from typing import Optional
from pathlib import Path
import cv2
import argparse
from anpr import LicensePlate_Detector, LicensePlate_OCR
from anpr.lp_detector import _ANPR_DIR_
from LPDGAN.LPDGAN import SwinTrans_G, LPDGAN_DEFALUT_CKPT_DIR
from imgproc_utils import normalize_brightness, L_CLAHE


def recognition_a_car(
    car_crop:np.ndarray, lp_detector:LicensePlate_Detector, 
    recong:LicensePlate_OCR, lpdgan:Optional[SwinTrans_G]=None, 
    return_lp_crop:bool=False
) -> tuple[list[int],tuple[str, str, float]]|tuple[list[int],tuple[str, str, float], np.ndarray]:
    """
    Args
    --
    - car_crop: the crop for the car that needs to detect license plate
    - lp_detector: license plate yolo detector
    - recog: OCR for license plate recognition
    - LPDGAN: deblur model. If no pass or give None, will performing OCR directly on raw crop with contrast enhancement
    
    Return
    --
    ```
    ([x0, y0, x1, y1], (ocr string, postprocessing string, conf of ocr))
    ```
    """
    license_plates_box = lp_detector(imgs=car_crop, conf=0.1)[0][0]
    lp = car_crop[
        license_plates_box[1]:license_plates_box[3], 
        license_plates_box [0]:license_plates_box[2]
    ].copy()
    lp = cv2.resize(lp, (224,112), interpolation=cv2.INTER_LINEAR)
    lp = normalize_brightness(lp)

    if lpdgan is not None:
        lp = lpdgan.inference(x=lp)
    
    txt, raw_txt, txt_conf = recong(crop=lp, preprocess_pipeline=L_CLAHE)
    
    return (license_plates_box, (txt, raw_txt, txt_conf), lp) if return_lp_crop \
        else (license_plates_box, (txt, raw_txt, txt_conf))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=Path)
    parser.add_argument("--lp_yolo", type=Path, default=_ANPR_DIR_/f"anpr_v8.pt")
    parser.add_argument("--deblur", action='store_true')
    parser.add_argument("--lpdgan", type=Path, default=LPDGAN_DEFALUT_CKPT_DIR/f"net_G.pt")
    
    args = parser.parse_args()
    
    print(f"loading license plate detection model {args.lp_yolo} ..")
    lp_detector = LicensePlate_Detector(model_path=args.lp_yolo)
    print("loading PaddleOCR model ..")
    recog = LicensePlate_OCR()
    lpdgan_deblur_model = None
    if args.deblur:
        print("using deblur", end=",")
        assert args.lpdgan.is_file()
        print("loading lpdgan", end=", ")
        lpdgan_deblur_model = SwinTrans_G(pretrained_ckpt=args.lpdgan, gpu_id=0)
        lpdgan_deblur_model.eval()
    print("All modules done.")

    print(f"read sample crop from {args.img}", end=" .. ")
    car_crop = cv2.imread(str(args.img))
    assert car_crop is not None

    print(f"start recongition ..")
    boxes, txt = recognition_a_car(
        car_crop=car_crop, 
        lp_detector=lp_detector, 
        recong=recog, 
        lpdgan=lpdgan_deblur_model
    )
    print(f"{args.img}: recog : {txt[0]} ({txt[1]}), {txt[2]}")



if __name__ == "__main__":
    
    main()
    

    