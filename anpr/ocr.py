import numpy as np
from paddleocr import PaddleOCR
import logging
from paddleocr.ppocr.utils.logging import get_logger
# from fast_plate_ocr import ONNXPlateRecognizer

_paddle_logger = get_logger()
_paddle_logger.setLevel(logging.ERROR)

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
        - pattern2: 000 0 AA (len = 6)
        - pattern3: 000 A AA (len = 6)
        - pattern4: 0AA 0 00 (len = 6)
        """

        ret = None
        if len(plate_number) > 6:
            # pattern1
            ret = ''.join([self._DIG2ENG.get(c, c) for c in plate_number[:3]])
            ret += ''.join([self._ENG2DIG.get(c, c) for c in plate_number[3:]])
        else :
            n_eng_pre3 = sum([1 for _ in plate_number[:3] if _.isalpha()])
            #print(plate_number[:3], n_eng_pre3)
            if n_eng_pre3 >= 2:
                # pattern4
                ret = ''.join(self._ENG2DIG.get(plate_number[0], plate_number[0]))
                ret += ''.join(self._DIG2ENG.get(c,c) for c in plate_number[1:3])
                ret += ''.join(self._ENG2DIG.get(c,c) for c in plate_number[3:])
            else:   
                # pattern2 & pattern3
                ret = ''.join([self._ENG2DIG.get(c, c) for c in plate_number[:3]]) 
                ret += plate_number[3]
                ret += ''.join([self._DIG2ENG.get(c, c) for c in plate_number[4:]])
        
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
            return 'None', 'None', conf
    
def character_eval(samples:list[tuple[str, str]]) -> tuple[float, float, float]:
    """
    Args
    ---
    - samples : [(pred0, gth0), (pred1, gth1), ... ]
    
    Returns
    ---
    three float: (precision, recall, f1_score) 
    """
    
    # Initialize total counts
    total_tp = 0  # Total True Positives
    total_fp = 0  # Total False Positives
    total_fn = 0  # Total False Negatives

    # Process each (predicted, ground_truth) pair
    for predicted, ground_truth in samples:
        # Initialize counts for the current sample
        tp = 0
        fp = 0
        fn = 0

        # Calculate TP, FP, FN for the current sample
        min_length = min(len(ground_truth), len(predicted))
        
        # Count True Positives (matching characters)
        for i in range(min_length):
            if ground_truth[i] == predicted[i]:
                tp += 1
            else:
                fp += 1
                fn += 1

        # Extra characters in predicted text are False Positives
        if len(predicted) > len(ground_truth):
            fp += len(predicted) - len(ground_truth)

        # Missing characters in predicted text are False Negatives
        elif len(ground_truth) > len(predicted):
            fn += len(ground_truth) - len(predicted)

        # Accumulate counts
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Calculate overall precision, recall, and F1 score
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    """
    # Print results
    print(f"Total True Positives (TP): {total_tp}")
    print(f"Total False Positives (FP): {total_fp}")
    print(f"Total False Negatives (FN): {total_fn}")
    print(f"Overall Precision: {precision:.3f}")
    print(f"Overall Recall: {recall:.3f}")
    print(f"Overall F1 Score: {f1_score:.3f}")
    """

    return precision, recall, f1_score

"""
m = ONNXPlateRecognizer('argentinian-plates-cnn-model')
x = m.run("submit_baseline/ccpb-006_北鎮派出所_2023-06-27T11_45_00+08_00/0.jpg")
print(x)

"""