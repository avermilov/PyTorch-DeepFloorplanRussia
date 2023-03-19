import gc
import os

import hydra
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from predict import post_process, room2rgb, boundary2rgb
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def balanced_entropy(preds, targets):
    eps = 1e-6
    m = nn.Softmax(dim=1)
    z = m(preds)
    cliped_z = torch.clamp(z, eps, 1 - eps)
    log_z = torch.log(cliped_z)
    num_classes = targets.size(1)
    ind = torch.argmax(targets, 1).type(torch.int)

    total = torch.sum(targets)

    m_c, n_c = [], []
    for c in range(num_classes):
        m_c.append((ind == c).type(torch.int))
        n_c.append(torch.sum(m_c[-1]).type(torch.float))

    c = []
    for i in range(num_classes):
        c.append(total - n_c[i])
    tc = sum(c)

    loss = 0
    for i in range(num_classes):
        w = c[i] / tc
        m_c_one_hot = F.one_hot((i * m_c[i]).permute(1, 2, 0).type(torch.long),
                                num_classes)
        m_c_one_hot = m_c_one_hot.permute(2, 3, 0, 1)
        y_c = m_c_one_hot * targets
        loss += w * torch.sum(-torch.sum(y_c * log_z, axis=2))
    return loss / num_classes


def cross_two_tasks_weight(rooms, boundaries):
    p1 = torch.sum(rooms).type(torch.float)
    p2 = torch.sum(boundaries).type(torch.float)
    w1 = torch.div(p2, p1 + p2)
    w2 = torch.div(p1, p1 + p2)
    return w1, w2


def BCHW2colormap(tensor, earlyexit=False):
    if tensor.size(0) != 1:
        tensor = tensor[0].unsqueeze(0)
    result = tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if earlyexit:
        return result
    result = np.argmax(result, axis=2)
    return result


def compare(images, rooms, boundaries, r, cw):
    image = (BCHW2colormap(images, earlyexit=True) * 255).astype(np.uint8)
    room = BCHW2colormap(rooms).astype(np.uint8)
    boundary = BCHW2colormap(boundaries).astype(np.uint8)
    r = BCHW2colormap(r).astype(np.uint8)
    cw = BCHW2colormap(cw).astype(np.uint8)
    room_post = post_process(r, cw).astype(np.uint8)

    f = plt.figure()
    plt.subplot(2, 3, 1)
    plt.title("image")
    plt.imshow(image)
    plt.subplot(2, 3, 2)
    plt.title("room gt")
    plt.imshow(room2rgb(room))
    plt.subplot(2, 3, 3)
    plt.title("bd gt")
    plt.imshow(boundary2rgb(boundary))
    plt.subplot(2, 3, 4)
    plt.title("r pred post")
    plt.imshow(room2rgb(room_post))
    plt.subplot(2, 3, 5)
    plt.title("r pred")
    plt.imshow(room2rgb(r))
    plt.subplot(2, 3, 6)
    plt.title("bd pred")
    plt.imshow(boundary2rgb(cw))
    return f


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    DEVICE = cfg.general.device

    run_dir = hydra.core.hydra_config.HydraConfig.get()["run"]["dir"]
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.mkdir(ckpt_dir)
    writer = SummaryWriter(log_dir=run_dir + "/logs/",
                           flush_secs=3)

    if cfg.general.seed is not None:
        seed_everything(cfg.general.seed)

    model = hydra.utils.instantiate(cfg.model)
    if cfg.general.model_ckpt is not None:
        model.load_state_dict(torch.load(cfg.general.model_ckpt), strict=False)
    model.to(DEVICE)

    OptimizerClass = hydra.utils.get_class(cfg.optimizer.optimizer_type)
    optimizer = OptimizerClass(model.parameters(), **cfg.optimizer.params)

    dataset = hydra.utils.instantiate(cfg.dataset.train)

    shuffle_type = cfg.general.shuffle_type
    shuffle_type = shuffle_type.lower().strip() if shuffle_type is not None else shuffle_type
    do_shuffle = shuffle_type == "standard"

    if "train_share" in cfg.dataset:
        print("Splitting dataset into train/val...")
        train_count = int(cfg.dataset.train_share * len(dataset))
        val_count = len(dataset) - train_count
        train_ds, val_ds = torch.utils.data.random_split(dataset, [train_count, val_count],
                                                         torch.Generator().manual_seed(cfg.dataset.split_seed))
        val_ds_list = [val_ds]
    else:
        print("Loading val dataset(s)...")
        train_ds = dataset
        val_ds_list = hydra.utils.instantiate(cfg.dataset.val)
        # val_ds = hydra.utils.instantiate(cfg.dataset.val)

    sampler = SubsetRandomSampler(indices=list(range(0, len(train_ds)))) if shuffle_type == "random" else None
    train_loader = DataLoader(train_ds, num_workers=cfg.general.num_workers,
                              batch_size=cfg.general.train_batch_size,
                              shuffle=do_shuffle, sampler=sampler)

    val_loader_list = [DataLoader(val_ds, shuffle=False, num_workers=cfg.general.num_workers,
                                  batch_size=1) for val_ds in val_ds_list]

    train_ds_size = len(train_loader.dataset)
    total_val_ds_size = sum([len(val_ds) for val_ds in val_ds_list])
    # val_ds_size = len(val_loader.dataset)

    scheduler = None
    scheduler_batch_wise_step = getattr(cfg.scheduler, "batch_wise_step", False)
    if cfg.scheduler is not None:
        SchedulerClass = hydra.utils.get_class(cfg.scheduler.scheduler_type)
        if scheduler_batch_wise_step:
            scheduler = SchedulerClass(optimizer, **cfg.scheduler.params,
                                       epochs=cfg.general.epochs, steps_per_epoch=len(train_loader))
        else:
            scheduler = SchedulerClass(optimizer, **cfg.scheduler.params)

    # additional loss coefficients
    room_w = cfg.loss.room_w
    boundary_w = cfg.loss.boundary_w

    # metric initializations
    PixelAccRoom = MulticlassAccuracy(num_classes=cfg.general.room_channels, average="micro").to(cfg.general.device)
    PixelAccCW = MulticlassAccuracy(num_classes=cfg.general.boundary_channels, average="micro").to(cfg.general.device)
    ClassAccRoom = MulticlassAccuracy(num_classes=cfg.general.room_channels, average="macro").to(cfg.general.device)
    ClassAccCW = MulticlassAccuracy(num_classes=cfg.general.boundary_channels, average="macro").to(cfg.general.device)

    for epoch in range(cfg.general.epochs):
        # TRAINING LOOP
        running_train_loss = .0
        running_train_room_loss = .0
        running_train_boundary_loss = .0
        running_train_pixel_acc_room = .0
        running_train_pixel_acc_cw = .0
        running_train_class_acc_room = .0
        running_train_class_acc_cw = .0

        for idx, (im, cw, r,) in tqdm.tqdm(enumerate(train_loader),
                                           total=len(train_loader), desc=f"Training #{epoch:03}"):
            im, cw, r = im.to(DEVICE), cw.to(DEVICE), r.to(DEVICE)
            # zero gradients
            optimizer.zero_grad()
            model.train()
            # forward
            logits_r, logits_cw = model(im)
            # loss
            loss1 = balanced_entropy(logits_r, r) * room_w
            loss2 = balanced_entropy(logits_cw, cw) * boundary_w
            w1, w2 = cross_two_tasks_weight(r, cw)
            loss = w1 * loss1 + w2 * loss2
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_train_loss += loss.item()
            running_train_room_loss += loss1.item()
            running_train_boundary_loss += loss2.item()

            pixel_acc_room = PixelAccRoom(logits_r, r.argmax(dim=1))
            pixel_acc_cw = PixelAccCW(logits_cw, cw.argmax(dim=1))
            class_acc_room = ClassAccRoom(logits_r, r.argmax(dim=1))
            class_acc_cw = ClassAccCW(logits_cw, cw.argmax(dim=1))

            running_train_pixel_acc_room += pixel_acc_room.item()
            running_train_pixel_acc_cw += pixel_acc_cw.item()
            running_train_class_acc_room += class_acc_room.item()
            running_train_class_acc_cw += class_acc_cw.item()

            writer.add_scalar("lr", get_lr(optimizer), global_step=epoch * len(train_loader) + idx)

            if scheduler is not None and scheduler_batch_wise_step:
                scheduler.step()

            # for name, value in {"batch_loss": loss.item(),
            #                     "batch_room_loss": loss1.item(),
            #                     "batch_boundary_loss": loss2.item()}.items():
            #     writer.add_scalar("train/" + name, value, global_step=epoch * len(train_loader) + idx)
            if idx % cfg.logging.log_train_every_n_epochs == 0:
                train_example = compare(im, r, cw, logits_r, logits_cw)
                writer.add_figure(f'train/image_{idx:03}', train_example, epoch)
        for name, value in {"epoch_loss": running_train_loss,
                            "epoch_room_loss": running_train_room_loss,
                            "epoch_boundary_loss": running_train_boundary_loss,
                            "epoch_pixel_acc_room": running_train_pixel_acc_room,
                            "epoch_pixel_acc_boundary": running_train_pixel_acc_cw,
                            "epoch_class_acc_room": running_train_class_acc_room,
                            "epoch_class_acc_boundary": running_train_class_acc_cw,
                            }.items():
            writer.add_scalar("train/" + name, value / train_ds_size, global_step=epoch)

        if scheduler is not None and not scheduler_batch_wise_step:
            print("STEP")
            scheduler.step()

        # TOTAL VALIDATION LOOP
        total_running_val_loss = .0
        total_running_val_room_loss = .0
        total_running_val_boundary_loss = .0

        total_running_pixel_acc_room = .0
        total_running_pixel_acc_cw = .0
        total_running_class_acc_room = .0
        total_running_class_acc_cw = .0
        with torch.inference_mode():
            model.eval()
            # SINGLE VALIDATION LOOP
            for val_loader in val_loader_list:
                val_ds_size = len(val_loader.dataset)
                running_val_loss = .0
                running_val_room_loss = .0
                running_val_boundary_loss = .0
                running_pixel_acc_room = .0
                running_pixel_acc_cw = .0
                running_class_acc_room = .0
                running_class_acc_cw = .0
                ds_name = val_loader.dataset.name
                for idx, (im, cw, r) in tqdm.tqdm(enumerate(val_loader),
                                                  total=val_ds_size, desc=f"Val {ds_name}"):
                    im, cw, r = im.to(DEVICE), cw.to(DEVICE), r.to(DEVICE)

                    optimizer.zero_grad()
                    # forward
                    logits_r, logits_cw = model(im)
                    # loss
                    loss1 = balanced_entropy(logits_r, r) * room_w
                    loss2 = balanced_entropy(logits_cw, cw) * boundary_w
                    w1, w2 = cross_two_tasks_weight(r, cw)
                    loss = w1 * loss1 + w2 * loss2
                    # ds running statistics
                    running_val_loss += loss.item()
                    running_val_room_loss += loss1.item()
                    running_val_boundary_loss += loss2.item()
                    # total running statistics
                    total_running_val_loss += loss.item()
                    total_running_val_room_loss += loss1.item()
                    total_running_val_boundary_loss += loss2.item()
                    # metrics
                    # predboundary = BCHW2colormap(logits_cw)
                    # predroom = BCHW2colormap(logits_r)

                    # print(torch.tensor(predroom).unsqueeze(0).type(torch.int32).shape, r.argmax(dim=1).shape)
                    pixel_acc_room = PixelAccRoom(logits_r, r.argmax(dim=1))
                    pixel_acc_cw = PixelAccCW(logits_cw, cw.argmax(dim=1))
                    class_acc_room = ClassAccRoom(logits_r, r.argmax(dim=1))
                    class_acc_cw = ClassAccCW(logits_cw, cw.argmax(dim=1))

                    running_pixel_acc_room += pixel_acc_room.item()
                    running_pixel_acc_cw += pixel_acc_cw.item()
                    running_class_acc_room += class_acc_room.item()
                    running_class_acc_cw += class_acc_cw.item()

                    total_running_pixel_acc_room += pixel_acc_room.item()
                    total_running_pixel_acc_cw += pixel_acc_cw.item()
                    total_running_class_acc_room += class_acc_room.item()
                    total_running_class_acc_cw += class_acc_cw.item()

                    # log batch
                    # for name, value in {
                    #     "batch_loss": loss.item(),
                    #     "batch_room_loss": loss1.item(),
                    #     "batch_boundary_loss": loss2.item(),
                    #     "batch_pixel_acc_room": pixel_acc_room.item(),
                    #     "batch_pixel_acc_boundary": pixel_acc_cw.item(),
                    #     "batch_class_pixel_acc_room": class_acc_room.item(),
                    #     "batch_class_pixel_acc_boundary": class_acc_cw.item()
                    # }.items():
                    #     writer.add_scalar(f"val_{ds_name}/{name}", value,
                    #                       global_step=epoch * len(train_loader) + idx)

                    # log image
                    if idx % cfg.logging.log_val_every_n_epochs == 0:
                        val_example = compare(im, r, cw, logits_r, logits_cw)
                        writer.add_figure(f'val_{ds_name}/image_{idx:03}', val_example, epoch)
                # log ds
                for name, value in {"epoch_loss": running_val_loss,
                                    "epoch_room_loss": running_val_room_loss,
                                    "epoch_boundary_loss": running_val_boundary_loss,
                                    "epoch_pixel_acc_room": running_pixel_acc_room,
                                    "epoch_pixel_acc_boundary": running_pixel_acc_cw,
                                    "epoch_class_acc_room": running_class_acc_room,
                                    "epoch_class_acc_boundary": running_class_acc_cw,
                                    }.items():
                    writer.add_scalar(f"val_{ds_name}/{name}", value / val_ds_size, global_step=epoch)
        # log total
        for name, value in {"epoch_loss": total_running_val_loss,
                            "epoch_room_loss": total_running_val_room_loss,
                            "epoch_boundary_loss": total_running_val_boundary_loss,
                            "epoch_pixel_acc_room": total_running_pixel_acc_room,
                            "epoch_pixel_acc_boundary": total_running_pixel_acc_cw,
                            "epoch_class_acc_room": total_running_class_acc_room,
                            "epoch_class_acc_boundary": total_running_class_acc_cw
                            }.items():
            writer.add_scalar(f"val/{name}", value / total_val_ds_size, global_step=epoch)
        val_loss = total_running_val_loss / total_val_ds_size
        if (epoch + 1) % cfg.logging.save_every_n_epochs == 0:
            torch.save(model.state_dict(), ckpt_dir + f"/model_epoch{epoch:03}_loss{val_loss:.0f}.pt")
        gc.collect()


if __name__ == "__main__":
    main()
