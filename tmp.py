import os
import shutil
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
path = '/home/artermiloff/PycharmProjects/PyTorch-DeepFloorplan/predict/input/42548963.jpg'
import pytesseract as pts
pts.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

res = pts.image_to_data(Image.open(path),
    lang='rus', output_type=pts.Output.DICT)
res_df = pd.DataFrame(res)
res_df = res_df[["top", "left", "width", "height", "conf", "text"]]
print(res_df.query("text != ''"))

import easyocr
reader = easyocr.Reader(["ru"])
result = reader.readtext(path)
for item in result:
    print(item)