import glob

import Levenshtein
from easyocr import Reader
import argparse
import cv2
from PIL import Image, ImageDraw, ImageFont


def cleanup_text(text):
    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV
    return "".join(
        [c if ord('а') <= ord(c) <= ord('я') or ord('А') <= ord(c) <= ord('Я') else "" for c in text]).strip()


print()
# path = '/home/artermiloff/PycharmProjects/PyTorch-DeepFloorplan/predict/input/147296.jpg'
word_dicts = {
    "room": ["комната", "кухня", "гостиная", "столовая", "кухнястоловая", "детская", "спальня",
             "кухнягостиная", "кабинет"],
    "bathroom": ["ванная", "санузел", "ванна"],
    "closet": ["шкаф", "гардероб", "гардеробная", "кладовая", "шкафкупе"],
    "hall": ["прихожая", "коридор", "холл", "тамбур"],
    "balcony": ["балкон", "лоджия", "веранда"],
    "trash": ["этаж", "общаяплощадь", "м", "владис", "общая"]
}
paths = glob.glob("/home/artermiloff/Datasets/FloorPlansRussiaSplit/ToLabelV3/*")
paths = [(int(path[path.rfind("/") + 1:-4]), path) for path in paths]
paths = [path for _, path in sorted(paths)]
for i, path in enumerate(paths):
    image = cv2.imread(path)
    reader = Reader(["ru"], gpu=True)
    results = reader.readtext(image)

    pil_image = Image.fromarray(image)
    img_draw = ImageDraw.Draw(pil_image)
    unicode_font = ImageFont.truetype("DejaVuSans.ttf", 10)
    for (bbox, text, prob) in results:
        init_text = text
        # display the OCR'd text and associated probability
        # unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        # cleanup the text and draw the box surrounding the text along
        # with the OCR'd text itself
        text = cleanup_text(text).lower().strip()
        # if text == 'м' or len(text) == 0:
        #     continue
        if "c" not in text and len(text) <= 2:
            continue
        print(path)
        print(f"Cleaned text: '{text}', initial: '{init_text}'", prob)
        min_dist, best_match = 1000, None
        for type, words in word_dicts.items():
            for word in words:
                distance = Levenshtein.distance(text, word)
                if distance < min_dist:
                    min_dist = distance
                    best_match = type
        if best_match == "trash" or min_dist >= 4:
            continue
        print('TEXT', text, best_match, min_dist)
        img_draw.text((tl[0] - 10, tl[1] - 10), best_match + f": '{text}'", fill=(255, 0, 0), font=unicode_font)
    pil_image.save(f"ocr/stricter_v2/test_{i:02}.png")

    print("\n")
    # cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    # cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_ITALIC, 0.8, (0, 255, 0), 2)
# show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)
