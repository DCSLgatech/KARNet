from enum import auto
import time
import pprint
import argparse
from cv2 import circle

import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader

from cdae.models.imitations import *
from cdae.datasets import TempAVImageSensorsSequenceDataset
from cdae.utils import *
from config.model.layers_config import *

pp = pprint.PrettyPrinter(indent=4)


def train(
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """ Train model for one epoch.

    Args:
        epoch (int): epoch number.
        model (torch.nn.Module): model to train.
        dataloader (torch.utils.data.Dataloader): dataloader.
        optimizer (torch.optim): optimizer.
        device (torch.device): device (cpu/gpu).
    
    Returns:
        loss_train (float): validation loss (numerical).
        Mean epoch accuracy (numerical).

    """

    model.train()
    loss_train = 0
    n_batches = len(dataloader)
    lr = get_lr(optimizer)
    accuracies = []

    tt = tqdm(dataloader, bar_format="|{bar:30}|{percentage:3.0f}%| {desc}")

    for batch_idx, batch in enumerate(tt):

        optimizer.zero_grad()

        a_out = model(batch[0].to(device))  
        a_gt = continous_to_discreet_temp(batch[1][:, -1, :].numpy()).to(device)

        loss, loss_num = calculate_loss(a_gt, a_out)
        loss_train += loss_num
        loss.backward()

        optimizer.step()

        acc = (torch.softmax(a_out, dim=1).argmax(dim=1) == a_gt).sum().float() / float(a_gt.size(0))
        accuracies.append(acc.detach().cpu())
        
        tt.set_description(f"Epoch {epoch:03}  lr={lr:.2e}  Batch: {(batch_idx+1):02}/{n_batches}  Loss: {loss_num:.4f}")

    loss_train = loss_train / n_batches

    return loss_train, np.mean(accuracies)


def validate(
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.cuda.device,
) -> Tuple[float, float]:
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

    loss_val = 0
    n_batches = len(dataloader)

    tt = tqdm(dataloader, bar_format="|{bar:30}|{percentage:3.0f}%| {desc}")

    with torch.no_grad():
        model.eval()
        accuracies = []

        for batch_idx, batch in enumerate(tt):
            
            a_out = model(batch[0].to(device))  
            a_gt = continous_to_discreet_temp(batch[1][:, -1, :].numpy()).to(device)

            _, loss_num = calculate_loss(a_gt, a_out)
            loss_val += loss_num

            acc = (torch.softmax(a_out, dim=1).argmax(dim=1) == a_gt).sum().float() / float(a_gt.size(0))
            accuracies.append(acc.detach().cpu())

            tt.set_description(f"Epoch {epoch:03} Batch: {(batch_idx+1):02}/{n_batches}  Loss: {loss_num:.4f}")

        loss_val = loss_val / n_batches

    return loss_val, np.mean(accuracies)


def main():

    parser = argparse.ArgumentParser(description="Imitation training - dynamic.")
    parser.add_argument("--config", default="config/model/carla_imitation.yaml", type=str, help="YAML config file path")
    args = parser.parse_args()
    config = load_config(args.config)
    config["logs"] = None

    # set_reproducibility_values(seed=config['random_seed'], deterministic=config['deterministic'])

    device = torch.device(config["device"])

    run_id          = generate_run_id()
    path_results    = prepare_run(run_id, header=config["log_header"])
    epoch_start     = 0

    print(run_id)
    path_log = f"results/{run_id}/{run_id}_log.txt"

    # Save config to results
    path_log = f"{path_results}/{run_id}_log.txt"
    save_config(config, f"{path_results}/{run_id}_config.yaml")
    
    autoencoder = get_autoencoder(config["autoencoder"], config["image_latent_size"])
    backbone = get_rnn(config, autoencoder)
    backbone.load_state_dict(torch.load(config["backbone_resume"]))

    model = DynamicLatentToActionTwo(backbone, config["image_latent_size"], config["sensor_size"]).to(device)

    dataset = TempAVImageSensorsSequenceDataset(
        path=config['path_train'], 
        config=config
    )

    dataset_val = TempAVImageSensorsSequenceDataset(
        path=config['path_validation'], 
        config=config
    )

    # Dataloaders
    dl_train = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True
    )

    dl_val = DataLoader(
        dataset_val, 
        batch_size=config["batch_size"], 
        num_workers=config["num_workers"],
        drop_last=True
    )

    lr = config["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["reg"])
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, config["lr_scheduler"])

    # Loss
    patience_lr = 0
    patience_es = 0
    prev_loss_val = 1e6
    prev_acc_val = 0
    prev_acc_tra = 0

    save_config(config, f"{path_results}/{run_id}_config.yaml")
    pprint.pprint(config)

    # Main loop
    for epoch in range(epoch_start, epoch_start + config["epochs"]):

        time_start = time.time()
        loss_train, acc_train = train(epoch, model, dl_train, optimizer, device)
        elapsed_train = time.time() - time_start
        loss_val, acc_val = validate(epoch, model, dl_val, device)
        elapsed_total = time.time() - time_start

        #scheduler.step()  # Scheduler

        #if loss_val < prev_loss_val:
        if acc_val > prev_acc_val:
            torch.save(model, f"{path_results}/{run_id}_best.pth")
            config["epoch_best"] = epoch
            patience_lr = 0
            patience_es = 0
            prev_loss_val = loss_val
            prev_acc_val = acc_val
            prev_acc_tra = acc_train
        else:
            patience_lr += 1
            patience_es += 1

        # Adjust lr if necessary
        if patience_lr >= config["patience_lr"]:
            patience_lr = 0
            del optimizer
            lr = lr * config["lr_alpha"]
            config["lr_last"] = lr
            #save_config(config, f"results/{run_id}/{run_id}_config.yaml")

            print(f"Adjusting lr to {lr}")

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config["reg"])

        print(f"Val   Epoch {epoch:03}  Elapsed: {elapsed_total:.2f}s \
            \nLoss (Tra): {loss_train:.4f} | Accuracy: {acc_train:.4f} | {elapsed_train:.2f} s \
            \nLoss (Val): {loss_val:.4f} | Accuracy: {acc_val:.4f} | {(elapsed_total-elapsed_train):.2f} s \
            \nLow  (Val): {prev_loss_val:.4f}, Accuracy (Tra/Val): {prev_acc_tra:.2f}/{prev_acc_val:.2f} Patience (lr/es) {patience_lr}/{patience_es}")

        # Save info
        info = dict({
            "epoch": epoch, 
            "lr": get_lr(optimizer), 
            "loss_train": loss_train, 
            "loss_val": loss_val, 
            "acc_train": acc_train, 
            "acc_val": acc_val
        })
        #pp.pprint(info)

        log_dict_to_csv(info, path_log)

        if patience_es >= config["patience_es"]:
            config["epoch_last"] = epoch
            print(f"Early stopping at epoch {epoch}")
            break
    
    save_config(config, f"{path_results}/{run_id}_config.yaml")
    torch.save(model, f"{path_results}/{run_id}_last.pth")
    print(f"Completed!")


if __name__ == "__main__":
    main()