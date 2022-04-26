import os
import pprint
import argparse
import time

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from pytorch_msssim import MS_SSIM

from cdae.utils import *
from cdae.models.autoencoders import *
from cdae.models.rnns import SimpleLatentSensorPredictionNet
from cdae.datasets import TempAVImageSensorsSequenceDataset
from config.model.layers_config import *


def calculate_loss(
    out: torch.Tensor, 
    target: torch.Tensor, 
    config: dict, 
    loss_msssim: MS_SSIM, 
    n_batch: int
)-> Tuple[torch.Tensor, np.ndarray]: 
    """ Single function to calculate loss both for train and validation.
    
    Args:
        out (torch.Tensor): model output.
        target (torch.Tensor): groundtruth.
        config (dict): configuration dictionary.
        loss_msssim (pytorch_msssim.MS_SSIM): loss function.
        n_batch (int): number of data points in the batch.

    Returns:
        Loss as torch tensor for backprop.
        Array (numpy) of loss component breakdown.
    """

    x_out, x_out_ae, x_out_lat, x_in_lat, s_out = out
    x_in, s_in = target
    xs = x_in.shape
    n = n_batch * config["no_seq"]

    # TODO: Optimize reshapes
    if config["last_image"]:
        # Using last prediction only for loss
        x_in_temp = x_in[:, -1, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
        x_out_temp = x_out[:, -1, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
        s_in_temp = s_in[:, -1, :]
        s_out_temp = s_out[:, -1, :]
    else:
        # Using all predicted frames for loss
        x_in_temp = x_in[:, 1:, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
        x_out_temp = x_out.reshape(-1, xs[2], xs[3], xs[4])
        s_in_temp = s_in[:, 1:, :]
        s_out_temp = s_out

    x_in_lat_temp = x_in_lat[:, 1:, :].reshape(-1, config["image_latent_size"])
    x_out_lat_temp = x_out_lat.reshape(-1, config["image_latent_size"])
    x_in_ae = x_in.reshape(-1, xs[2], xs[3], xs[4])  # Autoencoder input

    # Losses
    loss_image_1 = 1 - loss_msssim(x_out_ae, x_in_ae)  # AE loss (pre-LSTM)
    loss_image_2 = 1 - loss_msssim(x_out_temp, x_in_temp)  # AE loss (post-LSTM)

    loss_latent = F.smooth_l1_loss(x_out_lat_temp, x_in_lat_temp, reduction="sum") / n
    loss_sensors = F.smooth_l1_loss(s_out_temp, s_in_temp, reduction="sum") / n
    loss = (loss_image_1 + loss_image_2) / 2.0 + loss_latent + loss_sensors

    # No batch size division for MS-SSIM
    num_li = (
        loss_image_1.detach().cpu().numpy() + loss_image_2.detach().cpu().numpy()
    ) / 2.0
    num_ll = loss_latent.detach().cpu().numpy()
    num_ls = loss_sensors.detach().cpu().numpy()
    num_l = num_li + num_ll + num_ls

    loss_num = np.array([num_l, num_li, num_ll, num_ls])

    return loss, loss_num


def get_tt_desc(epoch, batch_idx, n_batches, lr, l):
    """ Temporary function to get tqdm description string.
    """
    s = f"Train Epoch {epoch:03}  Batch: {(batch_idx+1):02}/{n_batches}  LR {lr:.1e} \
                Loss: {l[0]:.6f} (I): {l[1]:.6f} (L): {l[2]:.6f} (S): {l[3]:.6f}"
    return s


def train(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_msssim: MS_SSIM,
    device: torch.device,
    config: dict,
) -> np.ndarray:
    """ Train model for one epoch.

    Args:
        epoch (int): epoch number.
        model (torch.nn.Module): model to train.
        dataloader (torch.utils.data.Dataloader): dataloader.
        optimizer (torch.optim): optimizer.
        loss_msssim (pytorch_msssim.MS_SSIM): MS-SSIM loss.
        device (torch.device): device (cpu/gpu).
        config (dict): config dictionary.
    
    Returns:
        loss_train (float): validation loss (numerical)

    """

    model.train()
    loss_train = np.zeros(4)
    n_batches = len(dataloader)
    lr = get_lr(optimizer)

    tt = tqdm(dataloader, bar_format="|{bar:30}|{percentage:3.0f}%| {desc}")

    for batch_idx, batch in enumerate(tt):
        optimizer.zero_grad()

        x_in, s_in = batch[0].to(device), batch[1].to(device)
        s_in[:, :, -1] = s_in[:, :, -1] / 23.0    # TODO: Temp Fix Speed Scaling
        x_out, x_out_ae, x_out_lat, x_in_lat, s_out = model(x_in, s_in)

        loss, loss_num = calculate_loss(
            (x_out, x_out_ae, x_out_lat, x_in_lat, s_out),
            (x_in, s_in),
            config,
            loss_msssim,
            n_batch=batch[0].shape[0],
        )

        tt.set_description(get_tt_desc(epoch, batch_idx, n_batches, lr, loss_num))
        loss_train += loss_num
        loss.backward()
        optimizer.step()

    loss_train = loss_train / n_batches

    return loss_train


def validate(
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    loss_msssim: MS_SSIM,
    device: torch.cuda.device,
    config: dict,
) -> np.ndarray:
    """ Validate model for one epoch.

    Args:
        epoch (int): epoch number.
        model (torch.nn.Module): model to train.
        dataloader (torch.utils.data.Dataloader): dataloader.
        loss_msssim (pytorch_msssim.MS_SSIM): MS-SSIM loss.
        device (torch.device): device (cpu/gpu).
        config (dict): config dictionary.
    
    Returns:
        loss_val (float): validation loss (numerical)
    """

    loss_val = np.zeros(4)
    n_batches = len(dataloader)

    tt = tqdm(dataloader, bar_format="|{bar:30}|{percentage:3.0f}%| {desc}")

    with torch.no_grad():

        model.eval()

        for batch_idx, batch in enumerate(tt):

            x_in, s_in = batch[0].to(device), batch[1].to(device)
            s_in[:, :, -1] = s_in[:, :, -1] / 23.0    # TODO: Temp Fix Speed Scaling
            x_out, x_out_ae, x_out_lat, x_in_lat, s_out = model(x_in, s_in)

            _, loss_num = calculate_loss(
                (x_out, x_out_ae, x_out_lat, x_in_lat, s_out),
                (x_in, s_in),
                config,
                loss_msssim,
                n_batch=batch[0].shape[0],
            )

            tt.set_description(get_tt_desc(epoch, batch_idx, n_batches, 0, loss_num))
            loss_val += loss_num

        loss_val = loss_val / n_batches

    return loss_val


def train_model(
    gpu: int, 
    args: argparse.Namespace, 
    config: dict
) -> None:
    """ Train model loop. Set as separate function for distributed.

    Args:
        gpu (int): GPU id
        args (Namespace): command line arguments
        config (dict): config loaded from yaml file.
    """

    # DistributedDataParallel
    if config["distributed"]:
        device = args.nr * args.gpus + gpu
        # Initialize the process group
        dist.init_process_group(
            backend="nccl", 
            init_method='env://',
            rank=device, 
            world_size=args.world_size
        )
    else:
        device = torch.device(config["device"])

    #set_reproducibility_values(seed=config['random_seed'], deterministic=config['deterministic'])
    #logs_tra, logs_val = get_dataset_splits_by_logs(len(os.listdir(config["path_dataset"])), config['train_split'], config['validation_split'])

    if "resume" in config:
        raise NotImplementedError
        # run_id = config["run_id"]
        # epoch_start = config["epoch_last"] + 1 if "epoch_last" in config else config["epoch_best"] + 1
        # lr = config["lr_last"]

        # model = torch.load(config["resume"], map_location="cpu").to(device)
        # print(f"Resumed {config['resume']}, epoch {epoch_start}, lr {lr}")

        # prev_loss_val = config["loss_val"]

    else:
        run_id = generate_run_id()
        epoch_start = 0
        lr = config["lr"]

        config["run_id"] = run_id
        config["lr_last"] = lr
        path_results = prepare_run(run_id, header=config["log_header"])

        prev_loss_val = 1e6

    print(run_id)
    path_log = f"{path_results}/{run_id}_log.txt"
    
    # TODO: Fix
    if "resume" in config:
        model = torch.load(config["resume"], map_location="cpu").to(device)
        print(f"Resumed {config['resume']}")

    else:
        
        # Load autoencoder
        if config["autoencoder_resume"] is None:
            autoencoder = None
        else:
            autoencoder = get_autoencoder(config["autoencoder"], config["image_latent_size"])
            autoencoder.load_state_dict(torch.load(config["autoencoder_resume"]))
            print(f"AE Warm Start {config['autoencoder_resume']}")

        # Load model
        model = SimpleLatentSensorPredictionNet(config, autoencoder, device=device).to(device)

    # Dataset
    ds_train = TempAVImageSensorsSequenceDataset(config["path_train"], config)
    ds_val = TempAVImageSensorsSequenceDataset(config["path_validation"], config)

    if config["distributed"]:
        model = DistributedDataParallel(model, device_ids=[gpu])
        sampler_train = DistributedSampler(ds_train, num_replicas=args.world_size, rank=device)
        sampler_val = DistributedSampler(ds_val, num_replicas=args.world_size, rank=device)
        shuffle=False
    else:
        sampler_train = None
        sampler_val = None
        shuffle=True

    # Dataloaders
    dl_train = DataLoader(ds_train, config["batch_size"], shuffle, sampler_train, num_workers=config["num_workers"])
    dl_val = DataLoader(ds_val, config["batch_size"], sampler_val, num_workers=config["num_workers"])

    # ---------- OPTIMIZER/SCHEDULER ----------
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["reg"])
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config['lr_scheduler'])

    # ---------- LOSS ----------
    n_channels = 1 if config["grayscale"] else 3
    loss_msssim = MS_SSIM(data_range=1, size_average=True, channel=n_channels)
    patience_lr = 0
    patience_es = 0

    save_config(config, f"{path_results}/{run_id}_config.yaml")
    pprint.pprint(config)

    # Main loop
    for epoch in range(epoch_start, epoch_start + config["epochs"]):

        time_start = time.time()
        loss_train = train(epoch, model, dl_train, optimizer, loss_msssim, device, config)
        elapsed_train = time.time() - time_start
        loss_val = validate(epoch, model, dl_val, loss_msssim, device, config)
        elapsed_total = time.time() - time_start

        if loss_val[0] < prev_loss_val:
            config["epoch_best"] = epoch
            config["loss_val"] = loss_val[0]

            torch.save(model, f"{path_results}/{run_id}_best.pth")
            save_config(config, f"{path_results}/{run_id}_config.yaml")

            patience_lr = 0
            patience_es = 0
            prev_loss_val = loss_val[0]
        else:
            patience_lr += 1
            patience_es += 1

        # Adjust lr if necessary
        if patience_lr >= config["patience_lr"]:
            patience_lr = 0
            del optimizer
            lr = lr * config["lr_alpha"]
            config["lr_last"] = lr
            save_config(config, f"{path_results}/{run_id}_config.yaml")
            print(f"Adjusting lr to {lr}...")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config["reg"])

        print(f"Val   Epoch {epoch:03}  Elapsed: {elapsed_total:.2f} s\
            \nLoss (Tra): {loss_train[0]:.6f} (I): {loss_train[1]:.6f} (L): {loss_train[2]:.6f} (S): {loss_train[3]:.6f} \t| {elapsed_train:.2f} s \
            \nLoss (Val): {loss_val[0]:.6f} (I): {loss_val[1]:.6f} (L): {loss_val[2]:.6f} (S) {loss_val [3]:.6f} \t| {(elapsed_total-elapsed_train):.2f} s \
            \nPatience (lr/es) {patience_lr}/{patience_es}")

        # Save info
        info = dict({
            'epoch': epoch, 
            'lr': get_lr(optimizer), 
            'loss_train': loss_train[0], 
            'loss_train_image': loss_train[1], 
            'loss_train_latent': loss_train[2],
            'loss_val': loss_val[0], 
            'loss_val_image': loss_val[1], 
            'loss_val_latent':loss_val[2],
            'loss_train_sen': loss_train[3], 
            'loss_val_sen': loss_val[3]
        })    

        log_dict_to_csv(info, path_log)

        if patience_es >= config["patience_es"]:
            config["epoch_last"] = epoch
            print(f"Early stopping at epoch {epoch}, validation loss hasn't decreased for {config['patience_es']} epochs.")
            break
            
    torch.save(model, f"{path_results}/{run_id}_last.pth")
    save_config(config, f"{path_results}/{run_id}_config.yaml")

    print(f"Completed!")


def main():

    parser = argparse.ArgumentParser(description="Autoencoder training.")
    parser.add_argument("--config", default="config/model/carla_rnn_sensor.yaml", type=str,help="YAML config file path")
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')
    args = parser.parse_args()
    config = load_config(args.config)

     # DistributedDataParallel
    if config["distributed"]:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '12355'    
        args.world_size = args.gpus * args.nodes  
        mp.spawn(train_model, nprocs=args.gpus, args=(args, config,))
        dist.destroy_process_group()
    else:
        train_model(0, args, config)    


if __name__ == "__main__":
    main()
