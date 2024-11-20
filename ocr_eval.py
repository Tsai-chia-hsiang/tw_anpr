from anpr import LicensePlate_OCR, OCR_Evaluator
import Levenshtein as lev
from typing import Optional
from LPDGAN.LPDGAN import SwinTrans_G, LPDGAN_DEFALUT_CKPT_DIR
from imgproc_utils import L_CLAHE, normalize_brightness
from pathlib import Path
import numpy as np
import cv2
import json
import os.path as osp
from tqdm import tqdm
import argparse
from argparse import Namespace
from imgproc_utils.vis import make_comparasion, make_canvas, make_text_card

def make_topk_vis_canvas(idxs, imgs:list, labels:list[str], pred:list[str], lp:LicensePlate_OCR, k=5, deblur_swintransformer:Optional[SwinTrans_G]=None) -> np.ndarray:
    c = [None]*k
    
    for i, idx in enumerate(idxs[:k]):
        src_img = cv2.imread(imgs[idx])
        db = deblur_swintransformer.inference(x=src_img) \
            if deblur_swintransformer is not None else \
            normalize_brightness(src_img)
        
        src_img = normalize_brightness(src_img)
        src_img = L_CLAHE(src_img)
        origin_txt = lp(src_img)[0]
        if origin_txt == 'n':
            origin_txt = ""
        od = lev.distance(origin_txt, labels[idx])
        db = L_CLAHE(db)

        c[i] = make_comparasion(
            origin=src_img, org_text=f"{origin_txt}({od})", 
            generate=db, gen_text=pred[idx]
        )
        c[i] = cv2.vconcat([
            make_text_card(text=labels[idx], card_width=c[i].shape[1]), 
            c[i]
        ])
    return make_canvas(c)
        

def read_json(file:str):
    with open(file, 'r') as fp:
        return json.load(fp)

def main(args:Namespace):
    
    title = "defalut"
    img_label:dict[str, str] = read_json(args.label_file)
    
    imgs = list(map(lambda x:args.data_root/x, list(img_label.keys())))
    labels = list(img_label.values())
    lp_ocr = LicensePlate_OCR()
    ocr_eva = OCR_Evaluator()
    
    deblur_swintransformer = None
    
    if args.deblur is not None:
        ckpt_path = args.deblur_ckpt_dir/args.deblur
        if ckpt_path.is_file():
            title=osp.sep.join(ckpt_path.parts[-2:])
        deblur_swintransformer = SwinTrans_G(
            pretrained_ckpt=args.deblur_ckpt_dir/args.deblur, 
            mode='inference', gpu_id=int(args.gpu),
            show_log=False
        )
    
    pred = [None] * len(imgs)
    
    for i, img_path in enumerate(tqdm(imgs)):
        src_img = cv2.imread(img_path)
        if deblur_swintransformer is not None:
            src_img = deblur_swintransformer.inference(x=src_img)
        else:
            if src_img.shape[:2] != (112, 224):
                src_img = cv2.resize(src_img, (224,112))
            src_img = normalize_brightness(src_img)
        
        plate_number = lp_ocr(crop=src_img, preprocess_pipeline=L_CLAHE)
        pred[i] = plate_number[0] if plate_number[0] != 'n' else ''
    
    acc = ocr_eva(pred=pred, gth=labels, method=args.eva, to_acc=True)
    print(f"{title},{acc}")
    
    """q = np.argsort(np.asarray(edit_distances))
    edit_distances = np.asarray(edit_distances)[q]
    vis_dir = Path("dataset/experiments")/f"{title}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    for i, qi in enumerate([0, 0.3, 0.6, 1]):
        idx= None
        if qi ==0:
            idx = q[:5]
        elif qi == 1:
            idx = q[-5:]
        else:
            pivot = int(len(q)*qi)
            idx = q[pivot-3:pivot+2]
        canvas = make_topk_vis_canvas(idxs=idx, imgs=imgs, labels=labels, pred=pred, lp=lp_ocr, deblur_swintransformer=deblur_swintransformer)
        cv2.imwrite(vis_dir/f"q{i}.png", canvas)"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--label_file", type=Path)
    parser.add_argument("--eva", type=str, default='lcs')
    parser.add_argument("--deblur_ckpt_dir", type=Path, default=LPDGAN_DEFALUT_CKPT_DIR)
    parser.add_argument("--deblur", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args = args)