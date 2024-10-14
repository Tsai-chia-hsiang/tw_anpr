from math import ceil
import cv2
import numpy as np

def make_text_card(text:str, card_height:int=40, card_width:int=70) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    # Create a black image
    text_card = np.zeros((card_height, card_width, 3), dtype=np.uint8)

    # Get the text size (width, height)
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Calculate the position to center the text
    x = (card_width - text_width) // 2
    y = (card_height + text_height) // 2  # Adjust y to account for text height

    # Put the text on the image (centered)
    cv2.putText(text_card, text, (x, y), font, font_scale, (255,255,255), font_thickness, cv2.LINE_AA)
    return text_card

def make_comparasion(origin:np.ndarray, org_text:str, generate:np.ndarray, gen_text:str)->np.ndarray:
    org_text_card = make_text_card(text=org_text, card_height=40, card_width=origin.shape[1])
    gen_text_card = make_text_card(text=gen_text, card_height=40, card_width=generate.shape[1])
    return cv2.vconcat([org_text_card, origin, gen_text_card, generate])

def make_canvas(s:list[np.ndarray], a_line:int=5, h_sep_pix:int=40, v_sep_pix:int=40):

    hsep = np.zeros((s[0].shape[0], h_sep_pix, 3), dtype=np.uint8)
    vsep = np.zeros((v_sep_pix, s[0].shape[1]*a_line + h_sep_pix*(a_line-1), 3), dtype=np.uint8)
    if len(s) < a_line:
        return cv2.hconcat([_ for i in range(len(s)) for _ in (s[i], hsep)][:-1])
    
    appending = ceil(len(s) / a_line) * a_line - len(s)
    s += [np.zeros_like((s[0])) for _ in range(appending)]
    canvas = None

    for i in range(0, len(s), a_line):
        this_line = cv2.hconcat([_ for j in range(a_line) for _ in (s[i+j], hsep)][:-1])
        canvas = cv2.vconcat([canvas, vsep, this_line]) if canvas is not None else this_line 
    return canvas
