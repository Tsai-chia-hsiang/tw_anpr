# ANPR for TW license plate

[English](./docs/readme_en.md)

## pipeline:
<img src="./docs/anplr.png">

## 執行方法:
1. 環境
Python >= 3.10
- 第三方套件
    - numpy
    - opencv 
    - torch
    - ultralytics
    - paddle 
        - cpu version : ```pip install paddlepaddle``` 
            - 這是我這邊使用的版本
        - gpu version : ```pip install paddlepaddle-gpu```
            - 如果使用 GPU 版本，請在執行前先確認  cuDNN libraries 路徑是否有在預設路徑裡面
            - 如果 cuDNN libraries 是沒有裝在 default 環境的話，請在執行 python 指令前，先將 cuDNN libraries 安裝的路徑添加到該 session 的 ```$LD_LIBRARY_PATH``` :
                - ```export LD_LIBRARY_PATH=/path/to/your/libraries:$LD_LIBRARY_PATH```
                - e.g.: 我將 cuDNN 額外用 conda 安裝在自己的環境，那執行前要先下: ```export LD_LIBRARY_PATH={$HOME}/.conda/envs/tenv/lib:$LD_LIBRARY_PATH``` ，然後再下 python ....
    - paddleocr:
        - ```pip install paddleocr```

    ** 第一次 inital paddleOCR 後，OCR 模型會被保存在:     
    - ```~/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer/en_PP-OCRv3_det_infer.tar```
    - ```~/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer/en_PP-OCRv4_rec_infer.tar```

2. 下載車牌偵測模型的權重: 請到[Automatic-Number-Plate-Recognition-Using-YOLOv8-EasyOCR/models](https://github.com/ANPR-ORG/Automatic-Number-Plate-Recognition-Using-YOLOv8-EasyOCR/tree/main/models) 下載 ```anpr_v8.pt```，儲存到 [```anpr/anpr_v8.pt```](```./anpr```)

3. (Optional) 下載 Pretrained Generator of LPDGAN 權重:
預訓練 lpdgan swintransformer 請到 https://drive.google.com/file/d/1sQD1uKOBpPCYGC8WGhoil47dOC2RjVQx/view?usp=sharing (```net_G.pt```)下載，並儲存到 [```./LPDGAN/checkpoints/net_G.pt```](./LPDGAN/checkpoints)

## 使用範例
請看參考 [```unit_inference.py```](./unit_inference.py)。

- 其中的 [```recognition_a_car()```](./unit_inference.py#L14) 有展示如何對一台車子的 crop 進行車牌辨識

[```unit_inference.py```](./unit_inference.py) 參數:
- ```--img``` : 要做車牌辨識的 crop 的路徑
- ```--lp_yolo``` : 車牌偵測的 yolov8 模型權重路徑。 
    - default 已經把路徑設置好了，如果要換其他權重再改
- ```--deblur``` : 如果要使用 LPGAN 去模糊，請下這個參數 (flag : store_true)
- ```lpdgan``` : 預訓練的 LPDGAN Generator的權重路徑。
    - **如果參數有下 --deblur，這個才會有用。**
    - default 已經把路徑設置好了，如果要換其他權重再改

針對該 crop，最終的車牌辨識結果是 [```recognition_a_car() 這個函數中的 txt 這個 str```](./unit_inference.py#L45) 這個變數

e.g.:
``` python unit_inference.py --img ./cars/0.png --deblur```

**以上指令 demo 用的圖片由於是隱私的資料及，如果真的想執行，請到 https://drive.google.com/file/d/1W7kjO5eJXpqG11BtDkuL0MsdQdxB7SL2/view?usp=sharing 要求權限。
- 請說明身分

## TODO
[TODO](./docs/TODO.md)

## Acknowledgments

This project utilizes code and resources from the following repositories:

- [LPDGAN](https://github.com/haoyGONG/LPDGAN.git)
    - origin paper: https://www.ijcai.org/proceedings/2024/0086.pdf
- [Automatic-Number-Plate-Recognition-Using-YOLOv8-EasyOCR](https://github.com/ANPR-ORG/Automatic-Number-Plate-Recognition-Using-YOLOv8-EasyOCR.git)

We deeply appreciate the work of these developers and their contributions to the open-source community.