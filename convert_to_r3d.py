import os
import shutil
import numpy as np
import glob
from PIL import Image
from validate_labeling import r3d_to_mask, mask_to_type, mask_to_r3d
from joblib import Parallel, delayed


def convert(mask_file):
    mask_name = mask_file[mask_file.rfind("/") + 1:]
    mask = np.asarray(Image.open(mask_file))
    if len(set(mask.flatten().tolist())) == 1:
        return
    values, counts = np.unique(mask, return_counts=True)
    if values.shape[0] == 1:
        return
    # print(mask.shape)

    close = mask.copy().tolist()
    for i in range(len(close)):
        for j in range(len(close[0])):
            close[i][j] = [255, 255, 255] if close[i][j] == 5 else [0, 0, 0]
    close_arr = np.array(close)

    wall = mask.copy().tolist()
    for i in range(len(wall)):
        for j in range(len(wall[0])):
            wall[i][j] = [255, 255, 255] if wall[i][j] == 0 else [0, 0, 0]
    wall_arr = np.array(wall)

    close_wall = mask.copy().tolist()
    for i in range(len(close_wall)):
        for j in range(len(close_wall[0])):
            close_wall[i][j] = [255, 255, 255] if close_wall[i][j] in [0, 5] else [0, 0, 0]
    close_wall_arr = np.array(close_wall)

    rooms = mask.copy().tolist()
    for i in range(len(rooms)):
        for j in range(len(rooms[0])):
            rooms[i][j] = [0, 0, 0] if rooms[i][j] in [0, 5, 6] else list(mask_to_r3d[rooms[i][j]])
    rooms_arr = np.array(rooms)

    multi = mask.copy().tolist()
    for i in range(len(multi)):
        for j in range(len(multi[0])):
            multi[i][j] = list(mask_to_r3d[multi[i][j]])
    multi_arr = np.array(multi)

    name = mask_name[:-4]
    # print(mask_file, mask_name, name)
    shutil.copy(f"/home/artermiloff/Downloads/224811_FloorPlansRussia(new shapes)/FloorPlansToLabel/img/{name}.jpg",
                f"dataset/floorplansrussia/{name}.jpg")
    Image.fromarray(close_arr.astype(np.uint8)).save(f"dataset/floorplansrussia/{name}_close.png")
    Image.fromarray(close_wall_arr.astype(np.uint8)).save(f"dataset/floorplansrussia/{name}_close_wall.png")
    Image.fromarray(multi_arr.astype(np.uint8)).save(f"dataset/floorplansrussia/{name}_multi.png")
    Image.fromarray(rooms_arr.astype(np.uint8)).save(f"dataset/floorplansrussia/{name}_rooms.png")
    Image.fromarray(wall_arr.astype(np.uint8)).save(f"dataset/floorplansrussia/{name}_wall.png")


path = "/home/artermiloff/Downloads/225893_FloorPlansRussia(new shapes)/FloorPlansToLabel/masks_machine/"
mask_files = sorted(glob.glob(path + "*"))
print(len(mask_files))
_ = Parallel(20)(delayed(convert)(file) for file in mask_files)
