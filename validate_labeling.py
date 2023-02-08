import glob
import os.path
import shutil
from collections import defaultdict

import PIL.Image
import networkx as nx
import numpy as np
from PIL import Image
from joblib import Parallel, delayed


# mask_to_type = {
#     0: "default_wall",
#     6: "background",
#     5: "window",
#     7: "room",
#     2: "bathroom",
#     1: "closet",
#     4: "balcony",
#     3: "hall",
#     9: "opening",
#     8: "wall",
#     10: "door",
#     11: "utility",
# }


# supervisely_rooms_to_mask = {
#     # (192, 192, 224): 1,  # "closet"
#     # (255, 60, 128): 5,  # "door/window"
#     # (0, 0, 0): 6,  # "background"
#     # (255, 255, 255): 0,  # "default_wall"
#     # (224, 255, 192): 7,  # "room"
#     # (255, 224, 128): 7,  # "room"
#     # (224, 224, 128): 8,  # "other"
#     # (224, 224, 224): 3,  # "hall"
#
#     (192, 255, 255): 2,  # "bathroom"
#     (224, 255, 192): 7,  # "room"
#     (255, 160, 96): 3,  # "hall"
#     (255, 224, 224): 4,  # "balcony"
#     (218, 222, 239): 1,  # closet
#     (0, 0, 0): 6,  # background
# }
#
# mask_to_supervisely_rooms = {k: v for k, v in supervisely_rooms_to_mask.items()}
#
# r3d_to_mask = {
#     (192, 192, 224): 1,  # "closet"
#     (255, 160, 96): 3,  # "hall"
#     (224, 255, 192): 7,  # "room"
#     (192, 255, 255): 2,  # "bathroom"
#     (255, 60, 128): 5,  # "door/window"
#     (255, 224, 224): 4,  # "balcony"
#     (0, 0, 0): 6,  # "background"
#     (255, 255, 255): 0,  # "default_wall"
#     (224, 255, 192): 7,  # "room"
#     (255, 224, 128): 7,  # "room"
#     (224, 224, 128): 8,  # "other"
#     (224, 224, 224): 3,  # "hall"
#     (142, 201, 148): 9,
#
#     # errors in masks
#     (1, 1, 1): 6,
#     (2, 2, 2): 6,
#     (3, 3, 3): 6,
#     (102, 102, 102): 6,
#     (119, 119, 119): 6,
#     (118, 118, 118): 6,
#     (153, 153, 153): 0,
#     (214, 214, 214): 0,
#     (218, 222, 239): 1
# }
#
# mask_to_r3d = {
#     1: (192, 192, 224),  # "closet"
#     3: (255, 160, 96),  # "hall"
#     7: (224, 255, 192),  # "room"
#     2: (192, 255, 255),  # "bathroom"
#     5: (255, 60, 128),  # "door/window"
#     4: (255, 224, 224),  # "balcony"
#     6: (0, 0, 0),  # "background"
#     0: (255, 255, 255),  # "default_wall"
#     7: (224, 255, 192),  # "room"
#     7: (255, 224, 128),  # "room"
#     8: (224, 224, 128),  # "other"
#     3: (224, 224, 224),  # "hall"
#
#     1: (218, 222, 239),  # closet
#     9: (142, 201, 148),  # "opening"
# }


# boundary_type_to_mask = {
#     "background": 0,
#     "wall": 1,
#     "window": 2,
#     "opening": 3,
#     "door": 4,
#     "utility": 5
# }
# room_type_to_mask = {
#     "background": 0,
#     "closet": 1,
#     "bathroom": 2,
#     "hall": 3,
#     "balcony": 4,
#     "room": 5,
#     "utility": 6,
# }
# type_to_mask = \
#     {'closet': 1, 'bathroom': 2, 'hall': 3, 'balcony': 4, 'window': 5, 'background': 6, 'room': 7, 'wall': 8,
#      'opening': 9, 'door': 10, 'utility': 11}







def ttm(k):
    return type_to_mask[k]


def mtt(k):
    return mask_to_type[k]


boundary_type_to_mask = {
    "background": 0,
    "wall": 1,
    "window": 2,
    "door": 3,
    "utility": 4,
    "openingtohall": 5,
    "openingtoroom": 6,
}
room_type_to_mask = {
    "background": 0,
    "closet": 1,
    "bathroom": 2,
    "hall": 3,
    "balcony": 4,
    "room": 5,
    "utility": 6,
    "openingtohall": 7,
    "openingtoroom": 8,
}
type_to_mask = {"defaultwall": 0, 'openingtohall': 1, 'openingtoroom': 2,
                'closet': 3,
                'bathroom': 4, 'hall': 5, 'balcony': 6,
                'window': 7, 'background': 8, 'room': 9,
                'wall': 10, 'opening': 11, 'door': 12, 'utility': 13}
mask_to_type = {v: k for k, v in type_to_mask.items()}


def validate_segmentation(path: str, min_comp_size: int = 50, r3d=False) -> (dict, str):
    image = np.array(Image.open(path))
    n, m = len(image), len(image[0])
    if len(set(np.asarray(image).flatten().tolist())) == 1:
        return {}, ""
    # if r3d:
    #     try:
    #         new_arr = [[0] * m for _ in range(n)]
    #         for i, j in np.ndindex(image.shape[:2]):
    #             pixel = tuple(image[i][j])
    #             new_arr[i][j] = r3d_to_mask[pixel]
    #         image = np.array(new_arr)
    #     except:
    #         print(path, pixel, i, j)
    # print(n, m, n * m)
    G = nx.Graph()
    G.add_nodes_from(range(0, n * m))

    def get_num(i, j):
        return m * i + j

    for i in range(n):
        for j in range(m):
            node_num = get_num(i, j)
            # left
            if 0 <= j - 1 < m:
                if image[i][j - 1] == image[i][j]:
                    neigh_num = get_num(i, j - 1)
                    G.add_edge(node_num, neigh_num)
            # right
            if 0 <= j + 1 < m:
                if image[i][j + 1] == image[i][j]:
                    neigh_num = get_num(i, j + 1)
                    G.add_edge(node_num, neigh_num)
            # up
            if 0 <= i - 1 < n:
                if image[i - 1][j] == image[i][j]:
                    neigh_num = get_num(i - 1, j)
                    G.add_edge(node_num, neigh_num)
            # down
            if 0 <= i + 1 < n:
                if image[i + 1][j] == image[i][j]:
                    neigh_num = get_num(i + 1, j)
                    G.add_edge(node_num, neigh_num)

    components = list(nx.connected_components(G))
    errors = path
    # print(res)
    d = defaultdict(int)
    for component in components:
        node_num = list(component)[0]
        comp_code = image[node_num // m][node_num % m]
        d[mask_to_type[comp_code]] += 1
        if len(component) < min_comp_size:
            errors += "\nvery small component detected: " + str(mask_to_type[comp_code])
        # print(mask_to_type[comp_code], len(component), round(len(component) * 100 / n / m, 1))

    # check if only 1 outside
    if d["background"] != 1: errors += "\nbackground count != 1?"
    if d["window"] < 1: errors += "\nno windows detected!"
    if d["door"] < 1: errors += "\nno doors detected!"
    if d["room"] < 1: errors += "\nno rooms detected!"
    if d["hall"] < 1: errors += "\nno halls detected?"
    if d["bathroom"] < 1: errors += "\nno bathrooms detected!"

    # assert

    return d, errors


import matplotlib.pyplot as plt

if __name__ == "__main__":
    paths = glob.glob(
        "/home/artermiloff/Downloads/226922_FloorPlansRussia(new shapes)/FloorPlansToLabel*/masks_machine/*")
    print(len(paths))
    results = Parallel(n_jobs=20)(delayed(validate_segmentation)(path, min_comp_size=25) for path in paths)
    # results = [validate_segmentation(path, min_comp_size=50) for path in paths]
    for i, (d, errors) in enumerate(results):
        # print(dict(d))
        if "\n" in errors:
            print(errors)
            plt.title(errors)
            plt.imshow(PIL.Image.open(errors[:errors.find("\n")].replace("machine", "human")))
            plt.show()

    # paths = glob.glob("dataset/r3d/newyork/t*/*_multi.png")
    # print(len(paths))
    # results = Parallel(n_jobs=20)(delayed(validate_segmentation)(path, min_comp_size=20, r3d=True) for path in paths)
    # schet = 0
    # for i, (d, errors) in enumerate(results):
    #     print(dict(d))
    #     if "\n" in errors:
    #         shutil.copy2(paths[i], os.path.join("r3d_bad/", paths[i][paths[i].rfind("/")+1:]))
    #         print(errors)
    #         if "small" in errors:
    #             schet += 1
    #
    # print(schet, len(paths))

    # 165    < 100
    # 137       < 50
