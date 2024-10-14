from argparse import ArgumentParser, Namespace
from pathlib import Path
import cv2
from anpr import Veh_Detector_with_LP_detection, LicensePlate_OCR
from LPDGAN.LPDGAN import SwinTrans_G, LPDGAN_DEFALUT_CKPT_DIR
from imgproc_utils import normalize_brightness, L_CLAHE
from imgproc_utils.vis import make_canvas, make_comparasion, make_text_card

def main(args:Namespace):

    lp_detector = Veh_Detector_with_LP_detection(gpuid=args.gpu_id)
    lpdgan = SwinTrans_G(pretrained_ckpt=args.lpdgan, gpu_id=args.gpu_id, mode="inference")
    ocr_model = LicensePlate_OCR()
    s = []
    lp_global = []
    for imgidx, img_pth in enumerate(args.infer_dir.glob("*.jpg")):
        frame = cv2.imread(img_pth)
        vis_image = frame.copy()
        dets = lp_detector(img=frame, lp_threshold=0.1, vehicle_crop=True)
        lp = []
        origin_lp = []
        for idx, v in enumerate(dets):
            x0, y0, xe, ye = v['vehicle_coors']
        
            if 'lp_coors' not in v:
                cv2.rectangle(vis_image, (x0, y0), (xe, ye), (225,232,220), thickness=1)
                continue
        
            x1, y1, x2, y2 = v['lp_coors']
            cv2.rectangle(vis_image, (x0, y0), (xe, ye), (110,132,46), thickness=1) 
            lp_shape = v['lp_crop'].shape[:2]

            
            vis_plate = cv2.resize(v['lp_crop'], (224, 112), interpolation=cv2.INTER_LINEAR)
            
            vis_plate = normalize_brightness(vis_plate)
            org_txt, r, _ = ocr_model(vis_plate)
            org_txt = f"{org_txt}({r})"
            clear = lpdgan.inference(x=v['lp_crop'])
            clear = L_CLAHE(clear)

            txt, r, _ = ocr_model(clear)
            txt = f"{txt}({r})"
            origin_lp.append(cv2.vconcat([make_text_card(text=org_txt, card_height=40, card_width=vis_plate.shape[1]), vis_plate]))
            lp.append(make_comparasion(origin=vis_plate, org_text=org_txt, generate=clear, gen_text=txt))
            
            vis_txt = make_text_card(f"{txt}", card_width=xe-x0)
            vis_image[y0: y0+vis_txt.shape[0], x0: x0 + vis_txt.shape[1]] = vis_txt
            clear = cv2.resize(clear, (lp_shape[1], lp_shape[0]), interpolation=cv2.INTER_CUBIC)
            vis_image[y1-clear.shape[0]:y1, x1:x2] = clear
            cv2.rectangle(vis_image, (x1, y1-clear.shape[0]), (x2, y1), (0,0,255), thickness=1)
            
        save_name = args.result_dir/img_pth.name
        assert cv2.imwrite(save_name, vis_image)
        cv2.imwrite(args.result_dir/f"lp_{img_pth.name}", make_canvas(s=lp, a_line=3))
        cv2.imwrite(args.result_dir/f"org_lp_{img_pth.name}", make_canvas(s=origin_lp, a_line=3))
        s.append(vis_image)
        lp_global += lp if imgidx != 1 else lp[:-1]
    cv2.imwrite(args.result_dir/f"deblur_cmp_canvas.jpg", make_canvas(s=lp_global, a_line=4))
    cv2.imwrite(args.result_dir/f"canvas.jpg", make_canvas(s=s, a_line=2))


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--infer_dir", type=Path, default=Path("dataset")/"sample_frames")
    parser.add_argument("--result_dir", type=Path, default=Path("result"))
    parser.add_argument("--lpdgan", type=Path, default=LPDGAN_DEFALUT_CKPT_DIR/f"net_G.pt")
    
    args = parser.parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    main(args=args)

