import csv
import glob
import os.path
import shutil

import PIL.Image
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import sys

from joblib import Parallel, delayed

from validate_labeling import r3d_to_mask, supervisely_rooms_to_mask, mask_to_supervisely_rooms

sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import *
import numpy as np


# def convert(mask_path, target_dir):
#     mask_name = mask_path[mask_path.rfind("/") + 1:]
#     mask = np.array(cv2.imread(mask_path))
#     if len(np.unique(mask)) == 1:
#         print(mask_path)
#         return None
#
#     h, w, _ = mask.shape
#     mask = np.max(mask, axis=2)
#     boundary = np.full((h, w), 69)
#     boundary[(mask == 1) | (mask == 2) | (mask == 3) | (mask == 4) | (mask == 6) | (mask == 7)] = 0  # background
#     boundary[(mask == 0) | (mask == 8)] = 1  # wall
#     boundary[mask == 5] = 2  # door/window
#     boundary[mask == 9] = 3  # opening
#     assert 69 not in boundary, f"not everything filled in boundary mask: {mask_path}"
#
#     room = np.full((h, w), 69)
#     room[(mask == 6) | (mask == 5) | (mask == 0) | (mask == 9) | (mask == 8)] = 0  # background
#     room[mask == 1] = 1  # closet
#     room[mask == 2] = 2  # bathroom
#     room[mask == 3] = 3  # hall
#     room[mask == 4] = 4  # balcony
#     room[mask == 7] = 5  # room
#     assert 69 not in room, f"not everything filled in room mask: {mask_path}"
#
#     boundary = boundary.flatten().astype(np.uint8)
#     room = room.flatten().astype(np.uint8)
#
#     img = np.asarray(PIL.Image.open(mask_path.replace("masks_machine", "img").replace("png", "jpg")))
#     h, w, c = img.shape
#     image = img.flatten().astype(np.uint8)
#
#     return {'filename': mask_name, 'image': list(image), 'boundary': list(boundary),
#             'room': list(room), 'door': '', "h": h, "w": w, "c": c}


def convert(mask_path, target_dir):
    mask_name = mask_path[mask_path.rfind("/") + 1:]
    mask = np.array(cv2.imread(mask_path))
    if len(np.unique(mask)) == 1:
        print(mask_path)
        return None

    h, w, _ = mask.shape
    mask = np.max(mask, axis=2)
    boundary = np.full((h, w), 69)
    boundary[(mask == 1) | (mask == 2) | (mask == 3) | (mask == 4) | (mask == 6) | (mask == 7)] = 0  # background
    boundary[(mask == 0) | (mask == 8)] = 1  # wall
    boundary[mask == 5] = 2  # window
    boundary[mask == 9] = 3  # opening
    boundary[mask == 10] = 4  # door
    boundary[mask == 11] = 5  # utility
    assert 69 not in boundary, f"not everything filled in boundary mask: {mask_path}"

    room = np.full((h, w), 69)
    room[(mask == 6) | (mask == 5) | (mask == 0) | (mask == 9) | (mask == 8) | (mask == 10)] = 0  # background
    room[mask == 1] = 1  # closet
    room[mask == 2] = 2  # bathroom
    room[mask == 3] = 3  # hall
    room[mask == 4] = 4  # balcony
    room[mask == 7] = 5  # room
    room[mask == 11] = 6  # utility
    assert 69 not in room, f"not everything filled in room mask: {mask_path}"

    PIL.Image.fromarray(room.astype(np.uint8)).save(os.path.join(target_dir, mask_name[:-4] + "_room.png"))
    PIL.Image.fromarray(boundary.astype(np.uint8)).save(os.path.join(target_dir, mask_name[:-4] + "_boundary.png"))

    # copy image
    img_src = mask_path.replace("masks_machine", "img").replace("png", "jpg")
    img_dst = os.path.join(target_dir, mask_name[:-4] + ".jpg")
    shutil.copy2(img_src, img_dst)


if __name__ == "__main__":
    mask_paths = sorted(
        glob.glob("/home/artermiloff/Downloads/226494_FloorPlansRussia(new shapes)/FloorPlansToLabel/masks_machine/*"))
    print(len(mask_paths))

    target_dir = "dataset/FPR_121_v2/"
    _ = Parallel(n_jobs=20)(delayed(convert)(path, target_dir) for path in mask_paths)
    # _ = [convert(mask_path, target_dir) for mask_path in mask_paths]
