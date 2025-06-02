import re
from typing import Callable

import cv2
import numpy as np
import numpy.typing as npt
from cnocr import CnOcr
from PIL import Image, ImageDraw, ImageFont


def check_chinese(text: str) -> bool:
    """判断字符串是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def draw_boxes_and_numbers(img, boxes, texts, base=1):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        # 使用较通用的字体
        font = ImageFont.truetype("Arial Unicode.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    for idx, (box, txt) in enumerate(zip(boxes, texts)):
        x1, y1, x2, y2 = map(int, box)
        # 画框
        draw.rectangle([x1, y1, x2, y2], outline='red', width=1)
        # 标号
        number = str(idx + base)
        # 获取文本的宽度和高度
        _, _, text_width, text_height = draw.textbbox((0, 0),
                                                      number,
                                                      font=font)
        # 画文本框
        draw.rectangle([x1 - text_width - 5, y1, x1, y1 + text_height + 5],
                       fill='red')
        draw.text((x1 - text_width - 3, y1), number, fill='white', font=font)

    return np.array(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))


def ocr_and_mark(image_path,
                 check: Callable[[str], bool] = check_chinese,
                 base=1) -> tuple[npt.NDArray, list[str]]:
    ocr = CnOcr(det_model_name='db_resnet18')
    img = cv2.imread(image_path)
    res = ocr.ocr(image_path)

    cn_texts = []
    cn_boxes = []
    for block in res:
        text = block['text']
        if check(text):
            cn_texts.append(text)
            # 坐标格式：[左上x, 左上y, 右下x, 右下y]
            box = block['position']
            # 部分版本坐标可能有4点，需转成[x1, y1, x2, y2]
            if len(box) == 4:  # 四个顶点
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                box = [min(xs), min(ys), max(xs), max(ys)]
            cn_boxes.append(box)

    marked_img = draw_boxes_and_numbers(img, cn_boxes, cn_texts, base=base)
    return marked_img, cn_texts
