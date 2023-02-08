import glob
import os
from typing import List

import torch

import albumentations as A
from PIL import Image

from importmod import *
import pandas as pd
import random
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader

from validate_labeling import boundary_type_to_mask, room_type_to_mask


class FillShape(nn.Module):
    def __init__(self, random=False):
        super().__init__()
        self.random = random

    def forward(self, image, boundary, room):
        h, w, c = image.shape
        if h == w:
            return image, boundary, room

        if h > w:
            pad_h_left, pad_h_right = 0, 0
            pad_w_left = (h - w) // 2 if not self.random else np.random.randint(0, h - w + 1)
            pad_w_right = h - w - pad_w_left
        else:
            pad_w_left, pad_w_right = 0, 0
            pad_h_left = (w - h) // 2 if not self.random else np.random.randint(0, w - h + 1)
            pad_h_right = w - h - pad_h_left

        padded_room = np.pad(room, ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right)), 'constant')
        padded_boundary = np.pad(boundary, ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right)), 'constant')
        padded_image = np.pad(image, ((pad_h_left, pad_h_right), (pad_w_left, pad_w_right), (0, 0)), 'constant',
                              constant_values=255)

        # print(padded_room, padded_boundary, padded_image.shape)
        return padded_image, padded_boundary, padded_room


def remap_values(arr, remap_dict):
    if remap_dict is None or len(remap_dict) == 0:
        return arr

    new_arr = arr.copy()
    for old_value, new_value in remap_dict.items():
        new_arr[arr == old_value] = new_value

    return new_arr


class r3dDataset(Dataset):
    def __init__(self, root_dir: str, num_boundary: int, num_room: int,
                 remap_room, remap_boundary, transform=None):
        # self.df = pd.read_csv(csv_file)
        # self.df2 = pd.concat([pd.read_csv('r3d2.csv'), pd.read_csv("floorplanrussia.csv")]).reset_index()

        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.num_boundary = num_boundary
        self.num_room = num_room
        self.remap_boundary = {boundary_type_to_mask[old]: new for old, new in remap_boundary.items()}
        self.remap_room = {room_type_to_mask[old]: new for old, new in remap_room.items()}

        # self.size = size
        self.transform = transform
        self.fill_shape_tsfm = FillShape(random=True)
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = np.asarray(Image.open(image_path))
        room = np.asarray(Image.open(image_path.replace(".jpg", "_room.png")))
        boundary = np.asarray(Image.open(image_path.replace(".jpg", "_boundary.png")))

        room = remap_values(room, self.remap_room)
        boundary = remap_values(boundary, self.remap_boundary)

        image, boundary, room = self.fill_shape_tsfm(image, boundary, room)

        if self.transform:
            tmp = self.transform(image=image, masks=[boundary, room])
            image, (boundary, room) = tmp["image"], tmp["masks"]

        image = self.to_tensor(image.astype(np.float32) / 255.0)
        boundary = self.to_tensor(F.one_hot(torch.LongTensor(boundary),
                                            self.num_boundary).numpy())
        # print(np.unique(room, return_counts=True))
        room = self.to_tensor(F.one_hot(torch.LongTensor(room),
                                        self.num_room).numpy())

        return image, boundary, room  # , door

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DFPdataset = r3dDataset(root_dir="dataset/FPR/",
                            num_boundary=4, num_room=6,
                            remap_room={2: 1, 3: 1, 4: 1, 5: 1, 7: 1},
                            remap_boundary=None,
                            transform=None)
    print(len(DFPdataset))
    for i in range(len(DFPdataset)):
        image, boundary, room = DFPdataset[i]
        print(i, image.shape)

    # for i in range(1):
    #     image, boundary, room = DFPdataset[8]
    #     print(image.shape, boundary.shape, room.shape)
    #
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(torchvision.transforms.ToPILImage()(image))
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(boundary)
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(room)
    #     # plt.subplot(2, 2, 4)
    #     # plt.imshow(door)
    #     plt.show()

    # breakpoint()

    gc.collect()
