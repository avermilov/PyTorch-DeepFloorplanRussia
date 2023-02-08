import gc
import sys

import torchvision.transforms

sys.path.append('./utils/')
from rgb_ind_convertor import *
from util import *
import cv2
from net import *
from data import *
import argparse
import matplotlib.pyplot as plt
import hydra
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import tqdm
from omegaconf import DictConfig


def BCHW2colormap(tensor, earlyexit=False):
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    result = tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if earlyexit:
        return result
    result = np.argmax(result, axis=2)
    return result


def initialize(args):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data
    trans = transforms.Compose([transforms.ToTensor()])
    orig = cv2.imread(args.image_path)
    orig = cv2.resize(orig, (512, 512))
    image = trans(orig.astype(np.float32) / 255.)
    image = image.unsqueeze(0).to(device)
    # model
    model = DFPmodel()
    model.load_state_dict(torch.load(args.loadmodel))
    model.to(device)
    return device, orig, image, model


def post_process(rm_ind, bd_ind):
    hard_c = (bd_ind > 0).astype(np.uint8)
    # region from room prediction 
    rm_mask = np.zeros(rm_ind.shape)
    rm_mask[rm_ind > 0] = 1
    # region from close wall line
    cw_mask = hard_c
    # regine close wall mask by filling the gap between bright line
    cw_mask = fill_break_line(cw_mask)

    fuse_mask = cw_mask + rm_mask
    fuse_mask[fuse_mask >= 1] = 255

    # refine fuse mask by filling the hole
    fuse_mask = flood_fill(fuse_mask)
    fuse_mask = fuse_mask // 255

    # one room one label
    new_rm_ind = refine_room_region(cw_mask, rm_ind)

    # ignore the background mislabeling
    new_rm_ind = fuse_mask * new_rm_ind

    return new_rm_ind


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def hex_to_rgb(s):
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


pred_boundary_type_to_rgb = {
    "background": hex_to_rgb("000000"),
    "wall": hex_to_rgb("FFFFFF"),
    "window": hex_to_rgb("FF3C80"),
    "door": hex_to_rgb("8020FF"),
    "utility": hex_to_rgb("4080E0"),
    "openingtohall": hex_to_rgb("0EF65F"),
    "openingtoroom": hex_to_rgb("417505"),
}
pred_room_type_to_rgb = {
    "background": hex_to_rgb("000000"),
    "closet": hex_to_rgb("C0C0E0"),
    "bathroom": hex_to_rgb("C0FFFF"),
    "hall": hex_to_rgb("FFA060"),
    "balcony": hex_to_rgb("FFE0E0"),
    "room": hex_to_rgb("E0FFC0"),
    "utility": hex_to_rgb("4080E0"),
    "openingtohall": hex_to_rgb("0EF65F"),
    "openingtoroom": hex_to_rgb("417505"),
}

room_int_to_rgb = {
    0: pred_room_type_to_rgb["background"],
    1: pred_room_type_to_rgb["hall"],
    2: pred_room_type_to_rgb["bathroom"],
    3: pred_room_type_to_rgb["utility"],
    4: pred_room_type_to_rgb["balcony"],
    5: pred_room_type_to_rgb["closet"],
    6: pred_room_type_to_rgb["room"],
}

boundary_int_to_rgb = {
    0: pred_boundary_type_to_rgb["background"],
    1: pred_boundary_type_to_rgb["wall"],
    2: pred_boundary_type_to_rgb["window"],
    3: pred_boundary_type_to_rgb["door"],
    4: pred_boundary_type_to_rgb["openingtohall"],
    # 5: pred_boundary_type_to_rgb["closet"],
    # 6: pred_boundary_type_to_rgb["room"],
}


def room2rgb(ind_im):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in room_int_to_rgb.items():
        rgb_im[(ind_im == i)] = rgb

    return rgb_im.astype(int)


def boundary2rgb(ind_im):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3))

    for i, rgb in boundary_int_to_rgb.items():
        rgb_im[(ind_im == i)] = rgb

    return rgb_im.astype(int)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DFPmodel(room_channels=args.room_channels,
                     boundary_channels=args.boundary_channels)
    model.load_state_dict(torch.load(args.weights_path))
    model.to(device)
    model.eval()

    src_files = glob.glob(args.src_dir + "/*")
    fill_shape_tsfm = FillShape(random=False)
    to_tensor = torchvision.transforms.ToTensor()
    with torch.inference_mode():
        for file in tqdm.tqdm(src_files):
            image = np.asarray(Image.open(file))
            h, w, c = image.shape
            orig = image.copy()

            image, _, _ = fill_shape_tsfm(image, np.zeros((h, w)), np.zeros((h, w)))
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
            image = to_tensor(image.astype(np.float32) / 255.0).unsqueeze(0)
            image = image.to(device)

            # run
            logits_r, logits_cw = model(image)
            predroom = BCHW2colormap(logits_r)
            predboundary = BCHW2colormap(logits_cw)

            # rgb_room_raw = ind2rgb(predroom, color_map=floorplan_fuse_map)
            # rgb_room = ind2rgb(post_process(predroom, predboundary), color_map=floorplan_fuse_map)

            rgb_room_raw = room2rgb(predroom)
            rgb_room = room2rgb(post_process(predroom, predboundary))
            rgb_boundary = boundary2rgb(predboundary)

            # plot
            # if False:
            # plt.figure(figsize=(8, 6), dpi=80)
            plt.subplot(1, 4, 1)
            plt.title("input")
            plt.imshow(orig[:, :, ::-1])
            plt.axis('off')
            plt.subplot(1, 4, 2)
            plt.title("room")
            plt.imshow(rgb_room_raw)
            plt.axis('off')
            plt.subplot(1, 4, 3)
            plt.title("room post")
            plt.imshow(rgb_room)
            plt.axis('off')
            plt.subplot(1, 4, 4)
            plt.title("boundary")
            plt.imshow(rgb_boundary)
            plt.axis('off')
            file_name = file[file.rfind("/") + 1:file.rfind(".")]
            plt.savefig(args.dst_dir + f"/{file_name}_grid.png", bbox_inches='tight', dpi=450)
            # plt.show()
            gc.collect()
            # grid_image = image_grid([orig, rgb_room, predboundary], rows=1, cols=3)
            # file_name = file[file.rfind("/") + 1:file.rfind(".")]
            # grid_image.save()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--weights_path', type=str,
                   default="/home/artermiloff/PycharmProjects/PyTorch-DeepFloorplan/experiment_logs/FPR_180_v1_all_rooms_smaller_cycle_rotate90_ADD_DOOR_UTILITY/train_20230205_102458/checkpoints/model_epoch059_loss595.pt")
    p.add_argument('--room_channels', type=int, default=7)
    p.add_argument('--boundary_channels', type=int, default=5)
    p.add_argument('--src_dir', type=str,
                   default="/home/artermiloff/PycharmProjects/PyTorch-DeepFloorplan/predict/input")
    p.add_argument('--dst_dir', type=str,
                   default="/home/artermiloff/PycharmProjects/PyTorch-DeepFloorplan/predict/output/all_rooms/")
    p.add_argument('--postprocess', action='store_true', default=True)
    args = p.parse_args()

    main(args)
