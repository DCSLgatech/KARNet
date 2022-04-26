import os 
import time
import pprint
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from pytorch_msssim import MS_SSIM
from tqdm import tqdm

from cdae.models.autoencoders import *
from cdae.datasets import TempAVImageDataset
from cdae.utils import *


def train(
    epoch: int, 
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim, 
    loss_msssim: MS_SSIM, 
    device: torch.device
) -> float:
    """ Train model for one epoch.

    Args:
        epoch (int): current epoch.
        model (torch.nn.Module): module currently being trained.
        dataloader (torch.utils.data.Dataloader): dataloader 
        optimizer (torch.optim): optimizer.
        loss_msssim (pytorch_msssim.MS_SSIM): MS-SSIM loss.
        device (torch.device): device.

    Returns:
        loss_train (float): training loss (numerical).
    """
 
    model.train()
    loss_train = 0
    n_batches = len(dataloader)
    
    lr = get_lr(optimizer)
    tt = tqdm(dataloader, bar_format="|{bar:30}| {percentage:3.0f}% | {desc}")     # TODO: Check

    for batch_idx, batch in enumerate(tt):
        optimizer.zero_grad()
        original = batch[0].to(device)
        reconstructed = model(original)

        loss = 1 - loss_msssim(reconstructed, original)
        #loss = F.mse_loss(original, reconstructed, reduction="mean")
        num_l = loss.detach().cpu().numpy() 
        loss_train += num_l
        loss.backward()
        optimizer.step()

        tt.set_description(f"Epoch {epoch:03}  lr={lr:.2e}  Batch: {(batch_idx+1):02}/{n_batches}  Loss: {num_l:.4f} \t {tt.format_dict['elapsed']:.2f} sec")

    loss_train = loss_train / n_batches

    return loss_train


def validate(
    epoch: int, 
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    loss_msssim: MS_SSIM, 
    device: torch.device
) -> float:
    """ Train model for one epoch.

    Args:
        epoch (int): current epoch.
        model (torch.nn.Module): module currently being trained.
        dataloader (torch.utils.data.Dataloader): dataloader
        loss_msssim (pytorch_msssim.MS_SSIM): MS-SSIM loss
        device (torch.device): device (cpu/gpu).

    Returns:
        loss_val (float): validation loss (numerical)
    """

    model.train()
    loss_val = 0
    n_batches = len(dataloader)
    
    tt = tqdm(dataloader, bar_format="|{bar:30}| {percentage:3.0f}% | {desc}")

    with torch.no_grad():

        model.eval()

        for batch_idx, batch in enumerate(tt):
            original = batch[0].to(device)
            reconstructed = model(original)
            loss = 1 - loss_msssim(reconstructed, original)
            #loss = F.mse_loss(original, reconstructed, reduction="mean")
            num_l = loss.detach().cpu().numpy() 
            loss_val += num_l

            tt.set_description(f"Epoch {epoch:03}  Batch: {(batch_idx+1):02}/{n_batches}  Loss: {num_l:.4f} \t {tt.format_dict['elapsed']:.2f} sec")

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

    # set_reproducibility_values(seed=config['random_seed'], deterministic=config['deterministic'])

    model = get_autoencoder(config["autoencoder"], config["image_latent_size"])

    if "resume" in config: 
        # TODO: Check
        _, filename = os.path.split(config['resume'])
        filename_data = filename.split('_')
        run_id = filename_data[0]
        epoch_start = int(filename_data[-1].split('.')[0])+1
        
        #model = torch.load(config["resume"], map_location="cpu").to(device)
        model.load_state_dict(torch.load(config["resume"]))
        print(f"Autoencoder resume {run_id}")
    else:
        epoch_start = 0
        run_id = generate_run_id()
        path_results = prepare_run(run_id, header = ['epoch', 'lr', 'loss_train', 'loss_val'])
        epoch_start = 0
        print(f"Autoencoder run {run_id}")

    model.to(device)
    
    ds_train = TempAVImageDataset(config['path_train'], None, config['cameras'], grayscale=config['grayscale'])
    ds_val = TempAVImageDataset(config['path_validation'], None, config['cameras'],grayscale=config['grayscale'])

    if config["distributed"]:
        model = DistributedDataParallel(model, device_ids=[gpu])
        sampler_train = DistributedSampler(ds_train, num_replicas=args.world_size, rank=device)
        sampler_val = DistributedSampler(ds_val, num_replicas=args.world_size, rank=device)
        shuffle=False
    else:
        sampler_train = None
        sampler_val = None
        shuffle=True
    
    dl_train = DataLoader(
        ds_train, 
        config['batch_size'], 
        shuffle, 
        sampler_train, 
        num_workers=config['num_workers'],
        drop_last=True
    )
    dl_val = DataLoader(ds_val, 
        config['batch_size'], 
        shuffle, 
        sampler_val, 
        num_workers=config['num_workers'],
        drop_last=True
    )

    lr = config["lr"]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config["reg"])
    loss_msssim = MS_SSIM(data_range=1, size_average=True, channel=1 if config['grayscale'] else 3)
    time_overall_start = time.time()

    path_log = f"{path_results}/{run_id}_log.txt"
    save_config(config, f"{path_results}/{run_id}_config.yaml")

    patience_lr = 0
    patience_es = 0
    prev_loss_val = 1e6

    for epoch in range(epoch_start, config["epochs"]):

        time_start = time.time()
        loss_train = train(epoch, model, dl_train, optimizer, loss_msssim, device)
        elapsed_train = time.time() - time_start
        loss_val = validate(epoch, model, dl_val, loss_msssim, device)
        elapsed_total = time.time() - time_start
        #scheduler.step()    # Scheduler

        if loss_val < prev_loss_val:
            #torch.save(model, f"{path_results}/{run_id}_best.pth")
            torch.save(model.state_dict(), f"{path_results}/{run_id}_state_dict_best.pth")
            config["epoch_best"] = epoch
            patience_lr = 0
            patience_es = 0
            prev_loss_val = loss_val
        else:
            patience_lr += 1
            patience_es += 1

        print(
            f"Epoch {epoch:03}  Elapsed: {elapsed_total:.2f}s, Tra: {loss_train:.6f}, Val: {loss_val:.6f}, Pat (lr/es): {patience_lr}/{patience_es}")
        info = dict({'epoch': epoch, 'lr': get_lr(optimizer), 'loss_train': loss_train, 'loss_val': loss_val})

        log_dict_to_csv(info, path_log)

        # Adjust lr if necessary
        if patience_lr >= config["patience_lr"]:
            patience_lr = 0
            del optimizer
            lr = lr * config["lr_alpha"]
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config["reg"])

        if patience_es >= config["patience_es"]:
            config["epoch_last"] = epoch
            print(f"Early stopping at epoch {epoch}")
            break

    plot_loss_ae(run_id)
    #torch.save(model, f"results/{run_id}/last.pth")
    torch.save(model.state_dict(), f"{path_results}/{run_id}_state_dict_last.pth")
    print(f"Total elapsed {(time.time() - time_overall_start):.2f} seconds")


def main():

    parser = argparse.ArgumentParser(description="Autoencoder training.")
    parser.add_argument("--config", default="config/model/carla_ae.yaml", type=str,help="YAML config file path")
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='Number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int, help='Number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')
    args = parser.parse_args()

    config = load_config(args.config)

    pprint.pprint(config)

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