from enum import auto
import os
import time
import pprint
import argparse

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import MS_SSIM
from tqdm import tqdm 

from cdae.models.rnns import *
from cdae.datasets import TempAVImageSequenceDataset
from cdae.utils import *


def calculate_loss(
    out: torch.Tensor, 
    target: torch.Tensor, 
    config:dict, 
    loss_msssim: MS_SSIM, 
    n_batch: int
) -> Tuple[torch.Tensor, np.ndarray]:
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

    x_out, x_out_ae, x_out_lat, x_in_lat = out
    x_in = target

    xs = x_in.shape
    n = n_batch * config["no_seq"]

    # TODO: Optimize reshapes
    if config["last_image"]:
        # Using last prediction only for loss
        x_in_temp  = x_in[:, -1, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
        x_out_temp = x_out[:, -1, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
    else:
        # Using all predicted frames for loss
        x_in_temp = x_in[:, 1:, :, :, :].reshape(-1, xs[2], xs[3], xs[4])
        x_out_temp = x_out.reshape(-1, xs[2], xs[3], xs[4])

    x_in_lat_temp = x_in_lat[:, 1:, :].reshape(-1, config['image_latent_size'])
    x_out_lat_temp = x_out_lat.reshape(-1, config['image_latent_size'])
    x_in_ae = x_in.reshape(-1, xs[2], xs[3], xs[4])  # Autoencoder input

    # Losses
    loss_image_1 = 1 - loss_msssim(x_out_ae, x_in_ae)
    loss_image_2 = 1 - loss_msssim(x_out_temp, x_in_temp)
    # loss_image = F.smooth_l1_loss(x_out_temp, x_in_temp, reduction='sum')
    loss_latent = F.smooth_l1_loss(x_out_lat_temp, x_in_lat_temp, reduction='sum') / n
    loss = (loss_image_1 + loss_image_2) / 2.0 + loss_latent

    # No batch size division for MS-SSIM
    num_li = (loss_image_1.detach().cpu().numpy() + loss_image_2.detach().cpu().numpy()) / 2.0
    num_ll = loss_latent.detach().cpu().numpy()
    num_l = num_li + num_ll

    return loss, np.array([num_l, num_li, num_ll])


def profiler_trace_ready() -> None:
    """ Temporary profile function. """
    torch.profiler.tensorboard_trace_handler("./log/rnn")
    print("Profiler trace ready")


def train(
    epoch: int, 
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim, 
    loss_msssim: MS_SSIM, 
    device: torch.device, 
    config: dict
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
    loss_train = np.zeros(3)       
    n_batches = len(dataloader) 
    lr = get_lr(optimizer)

    tt = tqdm(dataloader, bar_format="|{bar:30}|{percentage:3.0f}%| {desc}")

    # with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(
    #         wait=2,
    #         warmup=10,
    #         active=100,
    #         repeat=2),
    #         on_trace_ready=profiler_trace_ready(),
    #         record_shapes=True,
    #         #profile_memory=True,
    #         with_stack=True
    #     ) as profiler:

    for batch_idx, batch in enumerate(tt):
        optimizer.zero_grad()

        x_in = batch[0].to(device)
        x_out, x_out_ae, x_out_lat, x_in_lat = model(x_in)

        loss, loss_num = calculate_loss(
            (x_out, x_out_ae, x_out_lat, x_in_lat),
            x_in,
            config,
            loss_msssim,
            n_batch=batch[0].shape[0],
        )

        loss_train += loss_num

        loss.backward()
        optimizer.step()
            #profiler.step()

        tt.set_description(f"Train Epoch {epoch:03}  Batch: {(batch_idx+1):02}/{n_batches}  LR {lr:.1e}  Loss: {loss_num[0]:.6f} (I): {loss_num[1]:.6f} (L): {loss_num[2]:.6f}")

        #print(profiler.key_averages().table(sort_by='self_cpu_time_total', row_limit=20))

    loss_train = loss_train / n_batches
    return loss_train


def validate(
    epoch: int, 
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    loss_msssim: MS_SSIM, 
    device: torch.device, 
    config: dict
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
        loss_val (float): validation loss (numerical)

    """

    loss_val = np.zeros(3)
    n_batches = len(dataloader) 

    tt = tqdm(dataloader, bar_format="|{bar:30}|{percentage:3.0f}%| {desc}")

    with torch.no_grad():

        model.eval()

        for batch_idx, batch in enumerate(tt):

            x_in = batch[0].to(device)
            #xs = x_in.shape

            x_out, x_out_ae, x_out_lat, x_in_lat = model(x_in)

            _, loss_num = calculate_loss(
                (x_out, x_out_ae, x_out_lat, x_in_lat),
                x_in,
                config,
                loss_msssim,
                n_batch=batch[0].shape[0],
            )

            loss_val += loss_num

            tt.set_description(f"Valid Epoch {epoch:03}  Batch: {(batch_idx+1):02}/{n_batches}  Loss: {loss_num[0]:.6f} (I): {loss_num[1]:.6f} (L): {loss_num[2]:.6f}")

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
    Returns:
        None
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

    # Logging
    if "resume" in config:
        # TODO: Rewrite
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

    # ---------- MODEL LOADING ----------
    # RESUME BOTH AE AND RNN
    if "resume" in config:
        # TODO: Finish
        raise NotImplementedError

    # NEW RNN / NEW OR PRE-TRAINED AE
    else:
    
        if config["autoencoder_resume"] is None:
            # #TODO: Finish
            # raise NotImplementedError
            autoencoder = None
        else:
            # Load pre-trained autoencoder
            autoencoder = get_autoencoder(config["autoencoder"], config["image_latent_size"])
            autoencoder.load_state_dict(torch.load(config["autoencoder_resume"]))
            print(f"AE Warm Start {config['autoencoder_resume']}")
        
        model = get_rnn(config, autoencoder).to(device)

    # ---------- DATASET & DATALOADERS ----------
    ds_train = TempAVImageSequenceDataset(
        path=config['path_train'],
        logs = None,
        cameras = config['cameras'],
        grayscale=config["grayscale"],
        seq=config["no_seq"],
        delta=config["delta"]
    )

    ds_val = TempAVImageSequenceDataset(
        path=config['path_validation'],
        logs = None,
        cameras = config['cameras'],
        grayscale=config["grayscale"],
        seq=config["no_seq"],
        delta=config["delta"]
    )

    if config["distributed"]:
        model = DistributedDataParallel(model, device_ids=[gpu])
        sampler_train = DistributedSampler(ds_train, num_replicas=args.world_size, rank=device)
        sampler_val = DistributedSampler(ds_val, num_replicas=args.world_size, rank=device)
        shuffle=False
    else:
        sampler_train = None
        sampler_val = None
        shuffle=True

    dl_train = DataLoader(ds_train, config['batch_size'], shuffle, sampler_train, 
        num_workers=config['num_workers'], drop_last=True)
    dl_val = DataLoader(ds_val, config['batch_size'], shuffle, sampler_val, 
        num_workers=config['num_workers'], drop_last=True)

    # ---------- OPTIMIZER/SCHEDULER ----------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config['reg'])
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config['lr_scheduler'])

    # ---------- LOSS ----------
    n_channels = 1 if config["grayscale"] else 3
    loss_msssim = MS_SSIM(data_range=1, size_average=True, channel=n_channels)

    patience_lr = 0
    patience_es = 0

    save_config(config, f"{path_results}/{run_id}_config.yaml")
    pprint.pprint(config)

    # ---------- MAIN LOOP ----------
    for epoch in range(epoch_start, epoch_start + config['epochs']):

        time_start = time.time()
        loss_train = train(epoch, model, dl_train, optimizer, loss_msssim, device, config)
        elapsed_train = time.time() - time_start
        loss_val = validate(epoch, model, dl_val, loss_msssim, device, config)
        elapsed_total = time.time() - time_start

        #scheduler.step()    # Scheduler

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
            \nLoss (Tra): {loss_train[0]:.4f} (I): {loss_train[1]:.6f} (L): {loss_train[2]:.6f} \t| {elapsed_train:.2f} s \
            \nLoss (Val): {loss_val[0]:.4f} (I): {loss_val[1]:.6f} (L): {loss_val[2]:.6f} \t| {(elapsed_total-elapsed_train):.2f} s \
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
            'loss_train_sen': 0, 
            'loss_val_sen': 0
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

    # Can still use default_all.yaml config sinx_out_aece RNN is within the combined architecture
    parser = argparse.ArgumentParser(description="Autoencoder training.")
    parser.add_argument("--config", default="config/model/carla_rnn.yaml", type=str,help="YAML config file path")
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
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