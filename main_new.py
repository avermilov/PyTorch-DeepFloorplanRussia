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
    plt.title("bd pred post")
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

    else:
        print("Loading val dataset...")
        train_ds = dataset
        val_ds = hydra.utils.instantiate(cfg.dataset.val)

    sampler = SubsetRandomSampler(indices=list(range(0, len(train_ds)))) if shuffle_type == "random" else None
    train_loader = DataLoader(train_ds, num_workers=cfg.general.num_workers,
                              batch_size=cfg.general.train_batch_size,
                              shuffle=do_shuffle, sampler=sampler)

    val_loader = DataLoader(val_ds, shuffle=False, num_workers=cfg.general.num_workers,
                            batch_size=cfg.general.val_batch_size)

    train_ds_size = len(train_loader.dataset)
    val_ds_size = len(val_loader.dataset)

    scheduler = None
    scheduler_batch_wise_step = cfg.scheduler.batch_wise_step
    if cfg.scheduler is not None:
        SchedulerClass = hydra.utils.get_class(cfg.scheduler.scheduler_type)
        if scheduler_batch_wise_step:
            scheduler = SchedulerClass(optimizer, **cfg.scheduler.params,
                                       epochs=cfg.general.epochs, steps_per_epoch=len(train_loader))
        else:
            scheduler = SchedulerClass(optimizer, **cfg.scheduler.params)

    for epoch in range(cfg.general.epochs):
        running_train_loss = .0
        running_train_room_loss = .0
        running_train_boundary_loss = .0

        for idx, (im, cw, r,) in tqdm.tqdm(enumerate(train_loader)):
            im, cw, r = im.to(DEVICE), cw.to(DEVICE), r.to(DEVICE)
            # zero gradients
            optimizer.zero_grad()
            model.train()
            # forward
            logits_r, logits_cw = model(im)
            # loss
            loss1 = balanced_entropy(logits_r, r)
            loss2 = balanced_entropy(logits_cw, cw)
            w1, w2 = cross_two_tasks_weight(r, cw)
            loss = w1 * loss1 + w2 * loss2
            # backward
            loss.backward()
            # optimize
            optimizer.step()

            running_train_loss += loss.item()
            running_train_room_loss += loss1.item()
            running_train_boundary_loss += loss2.item()

            writer.add_scalar("lr", get_lr(optimizer), global_step=epoch * len(train_loader) + idx)

            if scheduler is not None and scheduler_batch_wise_step:
                scheduler.step()

            for name, value in {"batch_loss": loss.item(),
                                "batch_room_loss": loss1.item(),
                                "batch_boundary_loss": loss2.item()}.items():
                writer.add_scalar("train/" + name, value, global_step=epoch * len(train_loader) + idx)
            if idx % cfg.logging.log_train_every_n_epochs == 0:
                f1 = compare(im, r, cw, logits_r, logits_cw)
                writer.add_figure(f'train/image_{idx:03}', f1, epoch)

        if scheduler is not None and not scheduler_batch_wise_step:
            print("STEP")
            scheduler.step()

        for name, value in {"epoch_loss": running_train_loss / train_ds_size,
                            "epoch_room_loss": running_train_room_loss / train_ds_size,
                            "epoch_boundary_loss": running_train_boundary_loss / train_ds_size}.items():
            writer.add_scalar("train/" + name, value, global_step=epoch)

        running_val_loss = .0
        running_val_room_loss = .0
        running_val_boundary_loss = .0
        for idx, (im, cw, r) in tqdm.tqdm(enumerate(val_loader)):
            im, cw, r = im.to(DEVICE), cw.to(DEVICE), r.to(DEVICE)
            with torch.inference_mode():
                model.eval()
                optimizer.zero_grad()
                # forward
                logits_r, logits_cw = model(im)
                # loss
                loss1 = balanced_entropy(logits_r, r)
                loss2 = balanced_entropy(logits_cw, cw)
                w1, w2 = cross_two_tasks_weight(r, cw)
                loss = w1 * loss1 + w2 * loss2
            # statistics
            running_val_loss += loss.item()
            running_val_room_loss += loss1.item()
            running_val_boundary_loss += loss2.item()

            for name, value in {"batch_loss": loss.item(),
                                "batch_room_loss": loss1.item(),
                                "batch_boundary_loss": loss2.item()}.items():
                writer.add_scalar("val/" + name, value, global_step=epoch * len(train_loader) + idx)
            # if idx % 10 == 0:
            if idx % cfg.logging.log_val_every_n_epochs == 0:
                f2 = compare(im, r, cw, logits_r, logits_cw)
                writer.add_figure(f'val/image_{idx:03}', f2, epoch)

        for name, value in {"epoch_loss": running_val_loss / val_ds_size,
                            "epoch_room_loss": running_val_room_loss / val_ds_size,
                            "epoch_boundary_loss": running_val_boundary_loss / val_ds_size}.items():
            writer.add_scalar("val/" + name, value, global_step=epoch)
        val_loss = running_val_loss / val_ds_size
        if (epoch + 1) % cfg.logging.save_every_n_epochs == 0:
            torch.save(model.state_dict(),
                       ckpt_dir + f"/model_epoch{epoch:03}_loss{val_loss:.0f}.pt")
        gc.collect()


if __name__ == "__main__":
    main()
