import os
import shutil

import cv2
import numpy as np
import glob
import tqdm
import pandas as pd

from PIL import Image

# files = glob.glob("/home/artermiloff/Downloads/data_plan/*")
# for i in range(len(files) // 2000):
#     os.makedirs(f"/home/artermiloff/Datasets/FloorPlansRussiaSplit/part2_{i:02}/", exist_ok=True)
# for i, file in tqdm.tqdm(enumerate(files)):
#     part = i // 2000
#     dst = os.path.join(f"/home/artermiloff/Datasets/FloorPlansRussiaSplit/part2_{part:02}/",
#                        file[file.rfind("/") + 1:-5] + ".jpg")
#     try:
#         img = Image.open(file).convert("RGB").save(dst)
#     except:
#         print(i, file)

# 147296.jpg
# 42626239.jpg
#

# path = '/home/artermiloff/PycharmProjects/PyTorch-DeepFloorplan/predict/input/42548963.jpg'
# import pytesseract as pts
# pts.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
#
# res = pts.image_to_data(Image.open(path),
#     lang='rus', output_type=pts.Output.DICT)
# res_df = pd.DataFrame(res)
# res_df = res_df[["top", "left", "width", "height", "conf", "text"]]
# print(res_df.query("text != ''"))
#
# import easyocr
# reader = easyocr.Reader(["ru"])
# result = reader.readtext(path)
# for item in result:
#     print(item)

# import albumentations as A
# path = "/home/artermiloff/PycharmProjects/PyTorch-DeepFloorplan/dataset/FPR_211_v1/val_standard/10059044.jpg"
# img = cv2.imread(path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# for i in range(10):
#     # comp = A.ColorJitter(always_apply=True,
#     #                      brightness=0.5,
#     #                      contrast=0.5, saturation=0.5, hue=0.5)(image=img)["image"]
#     comp = A.JpegCompression(quality_lower=1, quality_upper=10, always_apply=True)(image=img)["image"]
#     cv2.imwrite(f"img_comp_{i:02}.jpeg", comp)
# print(img)
import glob, tqdm
from PIL import Image

files = glob.glob("/home/artermiloff/Datasets/FloorPlansRussiaSplit/ToLabelV3/*")

sch = 0
for file in files:
    img = Image.open(file)
    w, h = img.size
    if w != h:
        print(w, h, max(w/h, h/w))
        sch += 1
print(sch / len(files))
