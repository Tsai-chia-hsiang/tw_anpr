# ANPR for TW license plate

[中文](./readme.md)

## pipeline:
<img src="./docs/anplr.png">


## Execution
1. Enviroments
- Python >= 3.10
- Third-party Packages
    - numpy
    - opencv
    - torch
    - ultralytics
    - paddle
        - CPU version: ```pip install paddlepaddle``` 
            - recommended
        - GPU version: ```pip install paddlepaddle-gpu```
            - If using the GPU version, please ensure that the cuDNN libraries are located in the default path before running.
            - If the cuDNN libraries are not in the default environment, add the installation path to the session's `$LD_LIBRARY_PATH` before executing Python:
                - ```export LD_LIBRARY_PATH=/path/to/your/libraries:$LD_LIBRARY_PATH```
                - e.g.: If I installed cuDNN via conda in a custom environment, run: ```export LD_LIBRARY_PATH=${HOME}/.conda/envs/tenv/lib:$LD_LIBRARY_PATH```, then execute Python commands.

    - PaddleOCR: ```pip install paddleocr```

    ** After the initial PaddleOCR setup, OCR models will be saved at:

    - ```~/.paddleocr/whl/det/en/en_PP-OCRv3_det_infer.tar```
    - ```~/.paddleocr/whl/rec/en/en_PP-OCRv4_rec_infer.tar```

2. Download the License Plate Detection Model Weights:
Download the `anpr_v8.pt` file from the [Automatic-Number-Plate-Recognition-Using-YOLOv8-EasyOCR/models](https://github.com/ANPR-ORG/Automatic-Number-Plate-Recognition-Using-YOLOv8-EasyOCR/tree/main/models) and save it to [```anpr/anpr_v8.pt```](./anpr).

3. (Optional) Download the Pretrained Generator Weights for LPDGAN:
Download the pretrained LPDGAN SwinTransformer (`net_G.pt`) from [this link](https://drive.google.com/file/d/1sQD1uKOBpPCYGC8WGhoil47dOC2RjVQx/view?usp=sharing), and save it to [```./LPDGAN/checkpoints/net_G.pt```](./LPDGAN/checkpoints).


## Usage
Refer to [```unit_inference.py```](./unit_inference.py) for examples.

Function: [```recognition_a_car()```](./unit_inference.py#L14)
This function demonstrates how to recognize a car plate from a cropped image.

Parameters:
- ```--img```: Path to the cropped car image for license plate recognition.
- ```--lp_yolo```: Path to YOLOv8 model weights for license plate detection 
    - default set.
- ```--deblur```: Use LPGAN for deblurring (flag: store_true).
- ```--lpdgan```: Path to pre-trained LPDGAN generator weights 
    - default set
    - Please note that Providing only the `lpdgan` path will not suffice for deblurring. Deblurring will only be applied when the `--deblur` option is explicitly set. 
    

The license plate recognition result is refered to [```recognition_a_car() txt string variable```](./unit_inference.py#L45) string variable.

Example:
```python unit_inference.py --img ./cars/0.png --deblur```

**image of above demo command is from private data.** If you would like to run the demo, please request access at the following link: [Google Drive Link](https://drive.google.com/file/d/1W7kjO5eJXpqG11BtDkuL0MsdQdxB7SL2/view?usp=sharing).
- Please provide your identity and reason for requesting access.

## TODO
[TODO](./TODO.md)

## Acknowledgments

This project utilizes code and resources from the following repositories:

- [LPDGAN](https://github.com/haoyGONG/LPDGAN.git)
    - origin paper: https://www.ijcai.org/proceedings/2024/0086.pdf
- [Automatic-Number-Plate-Recognition-Using-YOLOv8-EasyOCR](https://github.com/ANPR-ORG/Automatic-Number-Plate-Recognition-Using-YOLOv8-EasyOCR.git)

We deeply appreciate the work of these developers and their contributions to the open-source community.