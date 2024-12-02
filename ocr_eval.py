from anpr import LicensePlate_OCR, OCR_Evaluator
from functools import partial
import shutil
import Levenshtein as lev
from typing import Optional
from LPDGAN.LPDGAN import SwinTrans_G
from imgproc_utils import L_CLAHE, normalize_brightness
from pathlib import Path
import numpy as np
import cv2
from LPDGAN.models.utils import load_networks
import torch
import json
from tqdm import tqdm
import argparse
from argparse import Namespace
from imgproc_utils.vis import make_comparasion, make_canvas, make_text_card
from LPDGAN.LPDGAN import SwinTrans_G

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
        
def load_g(ckpt:Path, dev:torch.device=torch.device("cpu"))->SwinTrans_G:

    if ckpt is None:
        return None
    if not ckpt.is_file():
        return None
    print(f"load from {ckpt}")
    db = SwinTrans_G()
    db_k = db.state_dict()
    sd = torch.load(ckpt, map_location='cpu')
    sd_exist = {k:v for k,v in sd.items() if k in db_k}
    db.load_state_dict(sd_exist)
    db.to(dev)
    db.eval()
    return db


def read_json(file:str):
    with open(file, 'r') as fp:
        return json.load(fp)

def main(args:Namespace):
    
    title = "defalut"
    img_label:dict[str, str] = read_json(args.label_file)
    db = load_g(ckpt=args.ckpt, dev=torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu"))
    if isinstance(db, SwinTrans_G):
        title = args.ckpt
    print(title)
    imgs = list(map(lambda x:args.data_root/x, list(img_label.keys())))
    labels = list(img_label.values())
    lp_ocr = LicensePlate_OCR()
    ocr_eva = OCR_Evaluator()

    pred = [None] * len(imgs)
    confs = []
    for i, img_path in enumerate(tqdm(imgs)):
        
        src_img = cv2.imread(img_path)

        if db is not None:
            src_img = db.inference(x=src_img)
        #else:
        #    src_img = normalize_brightness(src_img)
        #src_img = cv2.resize(src_img,dsize=None, fx=0.5, fy=0.5)
        plate_number = lp_ocr(crop=src_img)
        pred[i] = plate_number[1]
        confs.append(np.mean(plate_number[2]))
    
    acc, dist = ocr_eva(pred=pred, gth=labels, method='cer', to_acc=True, detail=True)
    
    print(f"{title},{acc}")
    R = Path("dataset/tw/cruiser")
    R.mkdir(parents=True, exist_ok=True)
    R_OK=R/"ok"
    R_OK.mkdir(parents=True, exist_ok=True)
    good_map = {}
    
    for d, i, l, c in zip(dist, imgs, labels, confs):
        if len(l) >= 5 and d == 0:            
            save = R_OK/l
            save.mkdir(parents=True, exist_ok=True)
            if l not in good_map:
                good_map[l] = 0

            shutil.copy(i, save/f"{good_map[l]}_{int(c*100)}.jpg")
            good_map[l] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path)
    parser.add_argument("--label_file", type=Path)
    parser.add_argument("--eva", type=str, default='lcs')
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args = args)