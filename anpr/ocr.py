import numpy as np
from paddleocr import PaddleOCR
import logging
from paddleocr.ppocr.utils.logging import get_logger
import Levenshtein as lev
# from fast_plate_ocr import ONNXPlateRecognizer

_paddle_logger = get_logger()
_paddle_logger.setLevel(logging.ERROR)

__all__ = ["LicensePlate_OCR", "cer", "all_correction_rate"]

class LicensePlate_OCR():
    
    def __init__(self) -> None:
        self._noise_chara = '.,: -'
        self.paddle_ocr_model = PaddleOCR(use_angle_cls=True, lang="en")
        self._ENG2DIG = {
            'O': "0",
            'I': "1",
            "Z": "2",
            "S": "5",
            "B": "8",
            "E": "8",
            "G": "6",
            "C": "6",
            "c": "6",
            "U": "0",
            "|": "1",
            "J": "7",
            "A": "4"
        }
        self._DIG2ENG = {
            "0":"O",
            "1":"I",
            "2":"Z",
            "4":"A",
            "5":"S",
            "6":"G",
            "7":"J",
            "8":"B"
        }

    def pattern_correction(self, plate_number:str) -> str:
        """
        - pattern1: AAA 0000 (len = 7)
        """
        ret = None
        if len(plate_number) > 6:
            # pattern1
            ret = ''.join([self._DIG2ENG.get(c, c) for c in plate_number[:3]])
            ret += ''.join([self._ENG2DIG.get(c, c) for c in plate_number[3:]])
        return ret
     
    def _count_area(self, d)->int:    
        if d is None:
            return 0
        return (d[0][1][0] - d[0][0][0])*(d[0][2][1] - d[0][0][1])

    def paddle(self, crop:np.ndarray) -> tuple[str, float]:
    
        result = self.paddle_ocr_model.ocr(crop, cls=True)[0]
        
        if result is None:
            return [None, -1.0]

        main_patch = np.argmax(np.array([self._count_area(r) for r in result]))
        return result[main_patch][1]
    
    def __call__(self, crop:np.ndarray, post_correction:bool=True) -> tuple[str, str, float]:

        raw_txt, conf = self.paddle(crop=crop)
        if raw_txt is not None:
            txt = None
            raw_txt = raw_txt.translate(str.maketrans('', '', self._noise_chara))
            if post_correction and len(raw_txt) > 5:
                txt = self.pattern_correction(plate_number=raw_txt)
            else:
                txt = raw_txt
            return txt, raw_txt, conf
        
        else:
            return 'n', 'n', conf
    

def cer(pred_gth: list[tuple[str, str]]) -> float:
    """
    Calculate the Character Error Rate (CER) for a list of predicted and ground truth string pairs.

    Args:
        pred_gth (list of tuple[str, str]): A list where each element is a tuple containing two strings
            - (prediction string, ground truth string).

    Returns:
        float: The Character Error Rate (CER) as a ratio.
    """

    edit_distance = 0
    N = 0

    for idx, (pred, gth) in enumerate(pred_gth):
        
        if len(gth) == 0:
            raise ValueError(f"Ground truth string of index {idx}: [{gth}] is empty")
    
        edit_distance += lev.distance(gth, pred)
        N += len(gth)
    

    return edit_distance / N if N > 0 else float('inf')

def all_correction_rate(pred_gth: list[tuple[str, str]]) -> tuple[list[int],float]:

    N = len(pred_gth)
    all_correct = [idx for idx, (pred, gth) in enumerate(pred_gth) if pred == gth]

    return all_correct, len(all_correct) / N
