import re
from typing import IO, Callable

import cv2
import numpy as np
import numpy.typing as npt
from cnocr import CnOcr
from PIL import Image, ImageDraw, ImageFont


def check_chinese(text: str) -> bool:
    """判断字符串是否包含中文"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def draw_boxes_and_numbers(img, boxes, texts, base=1, text_size=24):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    try:
        # 使用较通用的字体
        font = ImageFont.truetype("Arial Unicode.ttf", text_size)
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
        draw.rectangle([x1 - text_width - 2, y1, x1, y1 + text_height + 2],
                       fill='red')
        draw.text((x1 - text_width - 1, y1), number, fill='white', font=font)

    return np.array(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))


def ocr_and_mark(
    image_path: str | Image.Image | npt.NDArray | IO[bytes],
    check: Callable[[str], bool] = check_chinese,
    base=1,
    cnocr_kwargs: dict | None = None,
    text_size=24,
) -> tuple[npt.NDArray, list[str]]:
    ocr = CnOcr(**(cnocr_kwargs or {}))
    if isinstance(image_path, str):
        img = np.array(cv2.imread(image_path))
    elif isinstance(image_path, Image.Image):
        img = np.array(image_path)
    elif isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = np.array(Image.open(image_path))

    res = ocr.ocr(img)

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

    marked_img = draw_boxes_and_numbers(img,
                                        cn_boxes,
                                        cn_texts,
                                        base=base,
                                        text_size=text_size)
    return marked_img, cn_texts
