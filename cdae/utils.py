import os
import sys
import time
import random
import shutil
from pathlib import Path
from typing import Tuple
from functools import reduce
from datetime import date, datetime

import cv2
import yaml
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from cdae.models.autoencoders import *
from cdae.models.rnns import SimpleLatentPredictionNet
from config.model.layers_config import *


def count_parameters(model: torch.nn.Module) -> int:
    """ Get number of trainable model parameters. 
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7

    Args: 
        model (torch.nn.Module): model to count parameters.
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def set_reproducibility_values(seed: int, deterministic: bool = True) -> None:
    """ Reproducibility check. Sets random seeds.
        CUBLAS_WORKSPACE_CONFIG=:4096:2

    Args:
        seed (int): random seed torch/numpy/random.
        deterministic (bool): cuda deterministic algorithms flag.
            Sets cudnn becnhmark to False and force deterministic benchmark.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(deterministic) 
        

def get_dataset_splits_by_logs(n_logs, split_tra, split_val, str_logs="Log_") \
    -> Tuple[list, list, list]:
    """ Dataset split function. It has to be semi-custom in order to avoid having 
        overlapping sequences within a single log. Different logs are used for 
        training and validation.

    Args:
        n_logs (int): number of logs.
        split_tra (float): training split (0,1).
        split_val (float): validation split (0,1).
        str_logs (str): log folder name prefix, i.e. Log_
    Returns:
        logs_tra (list): list of folder names - training logs.
        logs_val (list): list of folder names - validation logs.
    """
    
    n_logs_train = int(n_logs*split_tra)
    n_logs_val = int(n_logs*split_val)
    logs_idx = np.arange(0, n_logs)
    random.shuffle(logs_idx)

    logs_tra = [f"{str_logs}{s}" for s in logs_idx[np.arange(0, n_logs_train)]]
    logs_val = [f"{str_logs}{s}" for s in logs_idx[np.arange(n_logs_train, n_logs_train+n_logs_val)]]
    logs_tst = [f"{str_logs}{s}" for s in logs_idx[np.arange(n_logs_train+n_logs_val, n_logs)]]

    return logs_tra, logs_val, logs_tst


def load_config(filename:str) -> dict:
    """ Loads .yaml config file. 
    
    Args: 
        filename (str): config filename.
    Returns:
        config (dict): config dictionary.
    """
    with open(filename) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    return config


def save_config(config: dict, path_config: str) -> None:
    """ Saves config to file. 
    
    Args: 
        config (dict): config dictionary
        path_config (str): destination path
    """

    if "autoencoder_config" in config:      # Delete autoencoder config from dictionary
        del config["autoencoder_config"]    

    with open(path_config, "w") as file_config:
        yaml.dump(config, file_config)


def get_dataset_splits(
    dataset: Dataset, train_split: float = 0.8, test_split: float = 0.1
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Obtain train/test/validation splits from a given Torch dataset.
    Args:
        dataset (torch.utils.data.Dataset): Source dataset to be split
        train_split (float): Train split ratio
        test_split (float): Test split ratio
    Returns:
        train_dataset (torch.utils.data.Subset): Training subset
        test_dataset (torch.utils.data.Subset): Testing subset
        val_dataset (torch.utils.data.Subset): Validation subset of size
            len(dataset) - len(dataset)*train_split - len(dataset)*test_split
    """

    n = len(dataset)
    train_size, test_size = int(train_split * n), int(test_split * n)
    val_size = n - train_size - test_size

    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size, val_size]
    )

    return train_dataset, val_dataset, test_dataset


def generate_run_id() -> str:
    """Generates run id number.
    
    Returns:
        Timestamp in sec (str).
    """
    return str(int(np.floor(datetime.timestamp(datetime.utcnow()))))


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Obtain learinng rate from the optimizer.
    
    Args:
        optimizer (torch.optim): optimizer
    Returns:
        Learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def prepare_run(run_id: str, header=None) -> None:
    """Create results folder. And populates header.
    
    Args:
        run_id (str): run id string.
        header (list[str]): list of header fields.
    Returns:
        Resulting path (str).
    """

    #path_results = os.path.join("results", run_id)
    path_results = os.path.join("results", f"{str(date.today())}_{run_id}")
    if not os.path.exists(path_results):
        os.mkdir(path_results)

    with open(os.path.join(path_results, f"{run_id}_log.txt"), "a+") as f:
        if header is not None:
            str_header = ""
            for h in header:
                str_header += f"{h}, "
            f.write(str_header[:-2] + "\n")

    return path_results


def log_dict_to_csv(info: dict, path: str) -> None:
    """ Log config dictionary to .csv file.

    Args:
        info (dict): dictionary with statistics
        path (str): destination path
    """

    with open(path, "a+") as f:
        str_info = "".join(['%s, ' % info[key] for key in info.keys()])
        f.write(str_info[:-2] + "\n")


def convert_velocity_raw_to_abs(path: str) -> None:
    """ Convert raw velocities obtained from .bag to absolute values.
        TODO: Refactor.

    Args:
        path (str): destination path.
    """

    data = pd.read_csv(os.path.join(path, "velocity_raw.csv"))
    vel = data[["field.vector.x", "field.vector.y", "field.vector.z"]].values
    abs_vel = np.linalg.norm(vel, axis=1)
    nsecs = data["%time"].values
    timestamp = nsecs.astype("<U16").reshape(-1,)
    df = pd.DataFrame(data=[timestamp, abs_vel], index=["time_stamp", "abs_vel"]).T
    df.to_csv(os.path.join(path, "velocity_abs.csv"), index=False)
    print("Completed")


def convert_velocity_to_json(path: str) -> None:
    """Convert raw velocities obtained from .bag to absolute values.
        TODO: Refactor.

    Args:
        path (str): destination path.
    """

    data = pd.read_csv(os.path.join(path, "velocity_raw.csv"))
    path_sensors = os.path.join(path, "sensors")
    vel = data[["field.vector.x", "field.vector.y", "field.vector.z"]].values
    abs_vel = np.linalg.norm(vel, axis=1)
    nsecs = data["%time"].values
    timestamp = nsecs.astype("<U16").reshape(-1,)

    # Create folder to store sensor data for each frame in  json format
    if not os.path.exists(path_sensors):
        os.mkdir(path_sensors)

    for t in timestamp:
        #print(timestamp)
        filename = os.path.join(path_sensors, f"{t}.json")
        data_sensors = {
            "vx": vel[t, 0],
            "vy": vel[t, 1],
            "vz": vel[t, 2],
            "v": abs_vel[t],
        }
        json_write(data_sensors, filename)

    print(f"{path_sensors} saved!")


def get_camera_timestamps(path: str) -> list:
    """Obtain a list of camera timestamps.
    Filenames are the filestamps.

    Args:
        path (str): path to the folder, e.g. .../Log1/FL
    Returns:
        timestamps(list): list of timestamps (int)
    """

    filenames = os.listdir(path)
    timestamps = [int(filename.split(".")[0]) for filename in filenames]
    return timestamps


def convert_sensors_to_json(
    path: str,
    folder_sensors: str,
    folder_camera_timestamps: str,
    vectorized=False,
    verbose=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw velocities stored in velocity_raw.csv to separate .json file.

    Args:
        path (str): path to the log
        folder_sensors (str): subfolder to save the sensor data
        folder_camera_timestamps (str): subfolder with camera timestamps,
            i.e. images to use as timestamp list, since rosbag data timestamps
            is not synchronized with camera shutters.
        vectorized (bool): vectorization (otherwise RAM error even with 32 gigs)
        verbose (bool): debug verbose
    Returns:
        vel (np.ndarray): velocity array (temporary)
        sync_error (np.ndarray): camera timestamp synchronization error
        ts_cam (np.ndarray): camera timestamps
    """

    # Create folder to store sensor data for each frame in json format
    path_sensors = os.path.join(path, folder_sensors)
    if not os.path.exists(path_sensors):
        os.mkdir(path_sensors)

    # Load data previously dumped from .bad to .csv
    data = pd.read_csv(os.path.join(path, "velocity_raw.csv"))

    ts_cam = get_camera_timestamps(os.path.join(path, folder_camera_timestamps))
    ts_cam = np.sort(np.array(ts_cam))
    ts_sen = data["%time"].values.astype("<U16").squeeze().astype("int")

    if verbose:
        print(f"ts_cam {ts_cam.shape}, ts_sen {ts_sen.shape}")

    closest = np.zeros(ts_cam.shape[0]).astype("int")

    # Calculate closest
    if vectorized:  # Bad, doesn't work for large dataset
        dist = np.abs(np.expand_dims(ts_cam, axis=1) - ts_sen)
        closest = dist.argmin(axis=1)
    else:
        for i, t in enumerate(ts_cam):
            dist = np.abs(t - ts_sen)
            closest[i] = dist.argmin()

    vel = data[["field.vector.x", "field.vector.y", "field.vector.z"]].values[
        closest, :
    ]
    vel_abs = np.linalg.norm(vel, axis=1)
    vel = np.concatenate([vel, np.expand_dims(vel_abs, axis=1)], axis=1)
    sync_error = np.abs(ts_cam - ts_sen[closest]) / 1e6

    for i, t in enumerate(ts_cam):
        filename = os.path.join(path_sensors, f"{t}.json")

        data_sensors = {
            "vx": vel[i, 0],
            "vy": vel[i, 1],
            "vz": vel[i, 2],
            "v": vel[i, 3],
        }
        json_write(data_sensors, filename)

    print(f"{path_sensors} saved!")

    return vel, sync_error, ts_cam


def convert_csv_sensors_to_np_udacity(
    path: str,
    filename: str,  # .csv filename
    data_field: str,  # data field name, e.g. 
    folder_camera_timestamps: str,
    verbose=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """ NEW - Udacity dataset

    Convert raw velocities stored in velocity_raw.csv to separate .json file.

    Args:
        path (str): path to the log
        folder_sensors (str): subfolder to save the sensor data
        folder_camera_timestamps (str): subfolder with camera timestamps,
            i.e. images to use as timestamp list, since rosbag data timestamps
            is not synchronized with camera shutters.
        vectorized (bool): vectorization (otherwise RAM error even with 32 gigs)
        verbose (bool): debug verbose
    Returns:
        vel (np.ndarray): velocity array (temporary)
        sync_error (np.ndarray): camera timestamp synchronization error
        ts_cam (np.ndarray): camera timestamps
    """

    # Load data previously dumped from .bad to .csv
    data = pd.read_csv(os.path.join(path, filename))
    filename_csv = os.path.splitext(filename)[0]

    ts_cam = get_camera_timestamps(os.path.join(path, folder_camera_timestamps))
    ts_cam = np.sort(np.array(ts_cam))
    ts_sen = data["%time"].values.astype("<U19").squeeze().astype("int")

    if verbose:
        print(f"ts_cam {ts_cam.shape}, ts_sen {ts_sen.shape}")

    closest = np.zeros(ts_cam.shape[0]).astype("int")

    # Vectorization eats RAM
    for i, t in enumerate(ts_cam):
            dist = np.abs(t - ts_sen)
            closest[i] = dist.argmin()
        
    field_values = data[data_field].values[closest]
    sync_error = np.abs(ts_cam - ts_sen[closest]) 
    np.save(f"{os.path.join(path, filename_csv)}", field_values)

    return sync_error, ts_cam


def aggregate_sensor_json(
    path: str, folder_sensors: str, subfolders_sensors: list
) -> None:
    """Concatenate different jsons into one.

    Args:
        path (str):
        folder_sensors (str):
        subfolder_sensors (str):
    Returns:
        None
    """

    path_subfolders = [
        reduce(os.path.join, [path, folder_sensors, subfolder])
        for subfolder in subfolders_sensors
    ]
    path_sensors = os.path.join(path, folder_sensors)

    ts = get_camera_timestamps(path_subfolders[0])

    for t in ts:
        data_sensors = dict()
        filename = f"{t}.json"
        for ps in path_subfolders:
            data_sensors.update(json_read(os.path.join(ps, filename)))

        json_write(data_sensors, os.path.join(path_sensors, filename))


def resize_image_folder(
    path_in: str, path_out: str, size: tuple, grayscale=True
) -> None:
    """Resizes all images in the folder.

    Args:
        path_in (str): Original image folder path.
        path_out (str): Resized image folder path.
        size (tuple): New image size
    """

    if not os.path.exists(path_out):
        os.mkdir(path_out)

    list_images = [
        os.path.join(path_in, f)
        for f in os.listdir(path_in)
        if f.endswith(".png") or f.endswith(".jpg")
    ]

    #ft = tqdm(list_images)

    for f in list_images:
        image = cv2.imread(f)

        if grayscale:
            image = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(path_out, os.path.split(f)[-1]), image)


def continous_to_discreet_temp(y: np.ndarray):
    """ Classification. Converts continuous throttle, steer, brake to 9 classes.
    TODO: Check and rewrite if necessary.

    Args:
        y (np.ndarray): array containing trottle, brake and 
    """
    steer = y[:, 1].copy()
    # Discretize
    steer[y[:, 1] > 0.05] = 2.0
    steer[y[:, 1] < -0.05] = 0.0
    steer[~np.logical_or(steer == 0.0, steer == 2.0)] = 1.0

    # Discretize throttle and brake
    throttle = y[:, 0]
    brake = y[:, 2]

    acc = brake.copy()
    acc[np.logical_and(brake == 0.0, throttle == 1.0)] = 2.0
    acc[np.logical_and(brake == 0.0, throttle == 0.5)] = 1.0
    acc[np.logical_and(brake == 1.0, throttle == 0.0)] = 0.0

    actions = np.vstack((acc, steer)).T
    # Convert actions to indices
    action_ind = actions[:, 0] * 3 + actions[:, 1]

    return torch.from_numpy(action_ind).long()


def temp_udacity_conversions(path_dataset: str) -> None:
    """ Temporary Udacity dataset conversion function.
    """

    #path_dataset = f"/mnt/storage0/Udacity/train/CH3_001_N"
    #path_dataset = f"/mnt/storage0/Udacity/train/CH03_002"
    #path_dataset = f"/mnt/storage0/Udacity/validation/CH3_001_S"
    path_sensors = f"{path_dataset}/sensors"

    sensor_files = ["steering_report", "brake_report", "throttle_report"]
    sensor_fields = ["field.steering_wheel_angle", "field.pedal_output", "field.pedal_output"]
    time_start = time.time()
    for filename, field in zip(sensor_files, sensor_fields):
        
        sync_error, ts_cam = convert_csv_sensors_to_np_udacity(
            path = path_dataset,
            filename=f"{filename}.csv",
            data_field=field,
            folder_camera_timestamps="center_camera",
            verbose=True
        )

    bins_thr = np.array([0, 0.155, 0.3])
    #bins_bra = np.array([0, 0.155, 0.3])   # Same threshold for brake
    steering_wheel_ratio = 14.8

    data_trottle = np.load(os.path.join(path_dataset, "throttle_report.npy"))
    data_brake = np.load(os.path.join(path_dataset, "brake_report.npy"))
    data_steering = np.load(os.path.join(path_dataset, "steering_report.npy"))

    data_trottle_q = (np.digitize(data_trottle, bins_thr) - 1.0) / 2.0

    #data_brake_q = (np.digitize(data_brake, bins_bra) - 1.0) / 2.0
    data_brake_q = np.copy(data_brake)
    data_brake_q[data_brake_q < 0.155] = 0
    data_brake_q[data_brake_q >= 0.155] = 1

    data_steering_n = data_steering / steering_wheel_ratio

    print(data_steering.shape, data_brake.shape, data_trottle.shape)

    if not os.path.exists(path_sensors):
        os.mkdir(path_sensors)

    for i, ts in enumerate(ts_cam):

        data_sensors = {
            "steering": data_steering[i], 
            "steering_normalized":  data_steering_n[i], 
            "trottle": data_trottle[i], 
            "throttle_quantized": data_trottle_q[i], 
            "brake": data_brake[i],
            "brake_quantized": data_brake_q[i]
        }

        filename = os.path.join(path_sensors, f"{ts}.json")
        json_write(data_sensors, filename)

    print(f"Elapsed {filename}: {(time.time() - time_start):.2f} seconds")


def convert_to_web(path_source:str, path_target:str, logs:list) -> None:
    """ Converts to WebDataset format - image data only (for autoencoder).

    Args:
        path_source (str): source folder
        path_target (str): target folder
        logs (list[str]): list of log folders
    Returns:
        None
    """
    n = len(logs)

    for i in tqdm(range(n)):
        path_f = reduce(os.path.join, [path_source, logs[i], "camera"])
        path_r = reduce(os.path.join, [path_source, logs[i], "camera_r"])
        path_l = reduce(os.path.join, [path_source, logs[i], "camera_l"])
        path_b = reduce(os.path.join, [path_source, logs[i], "camera_b"])

        path_s = reduce(os.path.join(path_source, logs[i], "sensors"))
        #print(path_f)

        files = os.listdir(path_f)

        for ff in files:
            basename, ext = os.path.splitext(ff)
            shutil.copy(os.path.join(path_f, ff), os.path.join(path_target, f"{basename}.f{ext}"))
            shutil.copy(os.path.join(path_r, ff), os.path.join(path_target, f"{basename}.r{ext}"))
            shutil.copy(os.path.join(path_l, ff), os.path.join(path_target, f"{basename}.l{ext}"))
            shutil.copy(os.path.join(path_b, ff), os.path.join(path_target, f"{basename}.b{ext}"))


# def covert_to_web_sink(dataset: Dataset, path: str) -> None:
#     """ Converts to WebDataset format.
#     TODO: Refactor.

#     Args: 
#         dataset (torch.utils.data.Dataset): dataset instance to convert.
#         path (str): destination path
#     """
#     sink = wds.TarWriter(path)

#     for index, (image, image_path) in enumerate(dataset):
#         if index%10000==0:
#             print(f"{index:8d}", end="\r", flush=True, file=sys.stderr)

#         pp = Path(image_path).parts

#         #image = np.moveaxis(image.numpy(), 0, -1)
#         image = image.numpy().squeeze()

#         sink.write({
#             "__key__": f"frame_{index:06d}",
#             "image.jpg": image,
#             "path": f"{pp[-3]}/{pp[-2]}/{pp[-1]}",
#         })

#     sink.close()


# utils_aux.py

def json_write(data: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def json_read(filename: str) -> dict:
    with open(filename, "r") as f:
        f.seek(0)
        data = json.load(f)
    return data


def plot_loss(
    data: dict,
    title: str = None,
    ylims: list = None,
    loss_test: np.ndarray = None,
    loss_sample: np.ndarray = None,
):
    """Plots loss."""
    fig = plt.figure(figsize=(21, 7))

    legend = ["Train", "Validation"]

    ax1 = fig.add_subplot(121)
    ax1.plot(data["epoch"], data["loss_train_latent"])
    ax1.plot(data["epoch"], data["loss_val_latent"])

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Smooth L1 Loss")
    ax1.title.set_text("Latent")
    ax1.grid(":")

    ax2 = fig.add_subplot(122)
    ax2.plot(data["epoch"], data["loss_train_v_abs"])
    ax2.plot(data["epoch"], data["loss_val_v_abs"])

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Smooth L1 Loss")
    ax2.title.set_text("Velocity")
    ax2.grid(":")

    if ylims is not None:
        ax1.set_ylim(ylims[0, :])
        ax2.set_ylim(ylims[1, :])

    if loss_sample is not None:
        ax1.scatter(data["epoch"][-1], loss_sample[1], s=20, marker="x")
        ax2.scatter(data["epoch"][-1], loss_sample[2], s=20, marker="x")
        legend.append("Sample")

    if loss_test is not None:
        ax1.scatter(data["epoch"][-1], loss_test[1], s=20, marker="x")
        ax2.scatter(data["epoch"][-1], loss_test[2], s=20, marker="x")
        legend.append("Test")

    ax1.legend(legend)
    ax2.legend(legend)

    fig.suptitle(title)

    return fig


def get_data_config(run_id: int, path_results: str):
    """Load config and data."""
    filename_log = os.path.join(path_results, f"{run_id}/{run_id}_log.txt")
    filename_config = os.path.join(path_results, f"{run_id}/{run_id}_config.yaml")
    data_lstm = np.genfromtxt(filename_log, dtype=float, delimiter=",", names=True)

    with open(filename_config) as file:
        config_lstm = yaml.load(file)

    return data_lstm, config_lstm


def plot_loss_4(run_id, path_results="results/", save_fig=True, text=""):
    """Plot total, latent, reconstruction and sensor losses."""

    data_lstm_1, config_lstm_1 = get_data_config(run_id, path_results)

    fig = plt.figure(figsize=(24, 8))
    fig.suptitle(text)

    ax1 = fig.add_subplot(221)
    # ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax1.plot(data_lstm_1["epoch"], data_lstm_1["loss_train"], alpha=0.8, c="g")
    ax1.plot(data_lstm_1["epoch"], data_lstm_1["loss_val"], alpha=0.8, c="b")

    ax1.legend(["Train", "Validation"])
    ax1.title.set_text("Loss - Total")
    #ax1.set_ylim([0, 1])

    ax1.plot()
    ax1.grid(':')

    ax2 = fig.add_subplot(222)
    ax2.set_ylabel("Loss")

    ax2.plot(data_lstm_1["epoch"], data_lstm_1["loss_train_latent"], alpha=0.8, c="g")
    ax2.plot(data_lstm_1["epoch"], data_lstm_1["loss_val_latent"], alpha=0.8, c="b")

    ax2.legend(["Train", "Validation"])
    ax2.title.set_text("Loss - Latent (S. L1)")
    # ax2.set_ylim([0,0.05])

    ax2.plot()
    ax2.grid(':')

    ax3 = fig.add_subplot(223)
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Loss")

    a = ax3.plot(
        data_lstm_1["epoch"], data_lstm_1["loss_train_image"], alpha=0.8, c="g"
    )
    b = ax3.plot(data_lstm_1["epoch"], data_lstm_1["loss_val_image"], alpha=0.8, c="b")

    ax3.legend(["Train", "Validation"])
    ax3.title.set_text("Loss - Reconstruction (MS-SSIM)")
    #ax3.set_ylim([0, 1])

    ax3.plot()
    ax3.grid(':')

    ax4 = fig.add_subplot(224)
    ax4.set_xlabel("Epochs")
    ax4.set_ylabel("Loss")

    ax4.plot(data_lstm_1["epoch"], data_lstm_1["loss_train_sen"], alpha=0.8, c="g")
    ax4.plot(data_lstm_1["epoch"], data_lstm_1["loss_val_sen"], alpha=0.8, c="b")

    ax4.legend(["Train", "Validation"])
    ax4.title.set_text("Loss - Sensor (S. L1)")
    # ax4.set_ylim([0,0.1])

    ax4.plot()
    ax4.grid(':')

    if save_fig:
        fig.savefig(os.path.join(path_results, f"{run_id}/{run_id}.png"), dpi=250)


def plot_loss_ae(run_id, path_results="results/", save_fig=True, ylim=None, text=""):
    """Plot total, latent, reconstruction and sensor losses."""

    filename_log = os.path.join(path_results, f"{run_id}/{run_id}_log.txt")
    data = np.genfromtxt(filename_log, dtype=float, delimiter=",", names=True)

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(111)

    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.tick_params(axis='both', which='minor', labelsize=16)

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax1.plot(data["epoch"], data["loss_train"], alpha=0.8, c="g")
    ax1.plot(data["epoch"], data["loss_val"], alpha=0.8, c="b")

    ax1.legend(["Train", "Validation"], prop={"size":20})
    ax1.title.set_text(f"Autoencoder, run_id {run_id}, {text}")

    if ylim is not None:
        ax1.set_ylim(ylim)

    ax1.plot()
    ax1.grid()

    if save_fig:
        fig.savefig(os.path.join(path_results, f"{run_id}/{run_id}.png"), dpi=250, transparent=True)


def get_autoencoder(type:str="simple", latent_size:int=128) -> torch.nn.Module:
    """ Loads autoencoder model of specified type and latent variable size.

    Args:
        type (str): "simple", "conv_att". Defaults to "simple".
        latent_size (int): latent variable size.
    Returns:
        autoencoder (torch.nn.Module): autoencoder model
    """

    config = {}
    
    if latent_size == 64:
        config["autoencoder_config"] = {
            "layers_encoder": layers_encoder_256_64, 
            "layers_decoder": layers_decoder_256_64,
        }
    elif latent_size == 128:
        config["autoencoder_config"] = {
            "layers_encoder": layers_encoder_256_128, 
            "layers_decoder": layers_decoder_256_128,
        }
    elif latent_size == 256:
        config["autoencoder_config"] = {
            "layers_encoder": layers_encoder_256_256, 
            "layers_decoder": layers_decoder_256_256,
        }
    
    if type == "simple":
        autoencoder = SimpleAutoencoder(latent_size, config["autoencoder_config"])
    elif type == "attention_latent":
        autoencoder = LatentAttentionAutoencoder(latent_size, config["autoencoder_config"])
    elif type == "attention_conv":
        config["autoencoder_config"]["layers_encoder"] = layers_encoder_256_128_c  
        autoencoder = CNNAttentionAutoencoder(latent_size, config["autoencoder_config"])
    else:
        raise NotImplementedError

    return autoencoder


def get_rnn(config:dict, autoencoder:torch.nn.Module=None):

    if autoencoder == None:
        autoencoder = get_autoencoder(config["autoencoder"], config["image_latent_size"])

    rnn = SimpleLatentPredictionNet(
        autoencoder=autoencoder,
        input_size=config['image_latent_size'], 
        hidden_size=config['image_latent_size'], 
        num_layers=config['lstm_n_layers'], 
        dropout=config['dropout'],
        device=torch.device("cpu"),
        rnn_arch=config["rnn_arch"]
    )

    return rnn