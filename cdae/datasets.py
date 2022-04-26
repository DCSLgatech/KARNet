import os
import re
from functools import reduce
from typing import List

import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image

from cdae.utils import json_read


class TempAVImageDataset(Dataset):
    
    """Class TempAVImageDataset. Single image per item."""
    def __init__(
        self,
        path: str,
        logs: List[str],
        cameras: List[str],
        transform=None,
        grayscale: bool = False,
    ):
        """
        Args:
            path (str): path to the dataset folder.
            logs (list): list of log folders.
            cameras (list): list of camera folders.
            transform (torchvision.transforms): transforms to apply to image.
            grayscale (bool): flag to load grayscale image.
        """

        self.path = path
        
        self.cameras = cameras
        self.grayscale = grayscale
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR
        self.file_list_images = []  # Construct file list (multiple folders)

        # Compose transforms
        if transform is not None:
            self.transform = transforms.Compose([
                transform,
                transforms.ToTensor(),
                #transforms.Normalize((0.54552912), (0.14597499))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.54552912), (0.14597499))        # Hardcoded for now
            ])

        if logs is not None:
            self.logs = logs
        else:
            self.logs = os.listdir(self.path)

        # TODO: Better loop
        for log in self.logs:
            for camera in cameras:
                path_camera = reduce(os.path.join, [path, log, camera])
                file_list = [
                    os.path.join(path_camera, f)
                    for f in os.listdir(path_camera)
                    if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
                ]
                file_list.sort(key=lambda f: int(re.sub('\D', '', f)))
                self.file_list_images += file_list

    def __len__(self):
        """Return dataset length."""
        return len(self.file_list_images)

    def __getitem__(self, index: int):
        """Get single image.
        
        Args:
            index (int): dataset index.
            
        Returns:
                (torch.Tensor): tensor containing the transformed image.
                (str): path to the image (needed for the inference comparison).
        """
        # Assuming that images are grayscale

        # image = cv2.imread(self.file_list_images[index], self.cv2_flag)
        # return self.transform(Image.open(self.image_files[index]).convert('L'))

        # image = Image.open(self.file_list_images[index]).convert('L')
        image = Image.open(self.file_list_images[index])

        # if self.grayscale:
        #     image = np.expand_dims(image, axis=0)
        return self.transform(image), self.file_list_images[index]
        #return image, self.file_list_images

    def get_filename(self, index):
        return self.file_list_images[index]


class LightningTempAVImageDataModule(pl.LightningDataModule):
    """ LightningDataModule for autoencoder.
    """

    def __init__(self, config, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        """
        """
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        
        self.path_train = config["path_train"]
        self.path_validatiopn = config["path_validation"]
        self.path_test = config["path_test"]


        self.cameras = config["cameras"]
        self.grayscale = config["grayscale"]
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.num_workers = config["num_workers"]

        if "transform" in config:
            raise NotImplementedError
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])


    def train_dataloader(self):
        dataloader_train = DataLoader(
            TempAVImageDataset(self.path_train, None, self.cameras, None, self.grayscale), 
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
        )
        return dataloader_train

    def val_dataloader(self):
        dataloader_validation = DataLoader(
            TempAVImageDataset(self.path_validatiopn, None, self.cameras, None, self.grayscale), 
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
        )
        return dataloader_validation

    def test_dataloader(self):
        dataloader_test = DataLoader(
            TempAVImageDataset(self.path, None, self.cameras, None, self.grayscale), 
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
        )
        return dataloader_test


    def __len__(self):
        """Return dataset length."""
        return len(self.file_list_images)


    # def __getitem__(self, index: int):
    #     """Get single image.
        
    #     Args:
    #         index (int): dataset index.
            
    #     Returns:
    #             (torch.Tensor): tensor containing the transformed image.
    #             (str): path to the image (needed for the inference comparison).
    #     """
    #     # Assuming that images are grayscale

    #     #image = cv2.imread(self.file_list_images[index], self.cv2_flag)
    #     #image = Image.open(self.file_list_images[index]).convert('L')
    #     image = Image.open(self.file_list_images[index])

    #     temp_dict = json_read(self.file_list_sensors[index])
    #     sensors = np.array([temp_dict[f] for f in self.fields])

    #     return self.transform(image), torch.from_numpy(sensors), \
    #         self.file_list_images[index], self.file_list_sensors[index]


    def get_filename(self, index):
        return self.file_list_images[index]


class TempAVImageSequenceDataset(Dataset):
    """Class TempAVImageSequenceDataset. Image sequences with specified length and delta. 
    
    """
    def __init__(
        self,
        path: str,
        logs: List[str],
        cameras: List[str],
        transform=None,
        seq: int = 5,
        delta: int = 1,
        grayscale: bool = False,
    ):
        """
        Args:
            path (str): path to the dataset folder.
            logs (list): list of log folders.
            cameras (list): list of camera folders.
            transform (torchvision.transforms): transforms to apply to image.
            seq (int): number of images in the sequence.
            delta (int): distance between consecutive frames.
            grayscale (bool): grayscale image flag.
        """

        self.path = path
        
        self.cameras = cameras
        self.seq = seq
        self.delta = delta
        self.seq_filelist = []  # List of sequence filenames
        self.grayscale = grayscale
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR

        if logs is not None:
            self.logs = logs
        else:
            self.logs = os.listdir(self.path)

        # Compose transforms
        if transform is not None:
            self.transform = transforms.Compose([
                transform,
                transforms.ToTensor(),
                # transforms.Normalize((0.4474), (0.3502))
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        # TODO: Better loop
        for log in self.logs:  # For each log
            for camera in cameras:  # For each camera

                path_camera = reduce(os.path.join, [path, log, camera])

                file_list_camera = [
                    os.path.join(path_camera, f)
                    for f in os.listdir(path_camera)
                    if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
                ]

                # Need to sort with regexp 
                file_list_camera.sort(key=lambda f: int(re.sub('\D', '', f)))

                n_camera = len(file_list_camera)  # Number of images per camera
                points = np.arange(delta * (self.seq - 1),
                                   n_camera - self.delta)  # Usable images
                seqs = [
                    np.arange(
                        points[idx] - self.delta * (self.seq - 1),
                        points[idx] + self.delta + 1,
                        self.delta,
                    ).astype(int) for idx in range(len(points))
                ]  # Image sequences indices

                for ss in seqs:
                    seq_files = [file_list_camera[s] for s in ss]
                    self.seq_filelist.append(seq_files)

        print("Dataset loaded")

    def __getitem__(self, index):
        """Get single image sequence. Note: no transform.
        
        Args:
            index (int): dataset index.
        
        Returns:
            images (torch.Tensor): tensor containing the sequence of images.
            sequence (list): list containing paths to sensor data of the corresponding 
                                images in the sequence.
        """

        # TODO: Add proper normalization
        sequence = self.seq_filelist[index]
        # images = np.array([cv2.imread(s, self.cv2_flag)
        #                    for s in sequence]) / 255.0
        # images = np.array([Image.open(self.file_list_images[index]).convert('L')
        #                    for s in sequence]) / 255.0
        
        #images = np.array([Image.open(s) for s in sequence]) / 255.0

        images = [self.transform(Image.open(s)) for s in sequence]

        # if self.grayscale:
        #     images = np.expand_dims(images, axis=3)
        # images = torch.FloatTensor(images)
        # images = images.permute(0, 3, 1, 2)

        #TODO: Check stack()
        return torch.stack(images), sequence

    def __len__(self):
        return len(self.seq_filelist)


class LightningTempAVImageSequenceDataModule(pl.LightningDataModule):
    """ LightningDataModule for autoencoder.
    """
    def __init__(self, config, train_transforms=None, val_transforms=None, test_transforms=None, dims=None):
        """
        """
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        self.path_train = config["path_train"]
        self.path_validatiopn = config["path_validation"]
        self.path_test = config["path_test"]

        self.cameras = config["cameras"]
        self.grayscale = config["grayscale"]
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.num_workers = config["num_workers"]

        self.no_seq = config["no_seq"]
        self.delta = config["delta"]

        if "transform" in config:
            raise NotImplementedError
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

    def train_dataloader(self):
        dataloader_train = DataLoader(
            TempAVImageSequenceDataset(
                path=self.path_train,
                logs = None,
                cameras = self.cameras,
                grayscale=self.grayscale,
                seq=self.no_seq,
                delta=self.delta
            ), 
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
        )
        return dataloader_train

    def val_dataloader(self):
        dataloader_validation = DataLoader(
            TempAVImageSequenceDataset(
                path=self.path_validatiopn,
                logs = None,
                cameras = self.cameras,
                grayscale=self.grayscale,
                seq=self.no_seq,
                delta=self.delta
            ), 
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
        )
        return dataloader_validation

    def test_dataloader(self):
        dataloader_test = DataLoader(
            TempAVImageSequenceDataset(
                path=self.path_test,
                logs = None,
                cameras = self.cameras,
                grayscale=self.grayscale,
                seq=self.no_seq,
                delta=self.delta
            ), 
            num_workers=self.num_workers,
            batch_size=self.batch_size, 
        )
        return dataloader_test

class TempAVImageSensorsSequenceDataset(TempAVImageSequenceDataset):
    """Class TempAVImageSensorsSequenceDataset. Image sequences"""
    def __init__(
        self,
        path: str,
        config,
        sensors: str = "sensors",
        transform=None,
    ):
        """
        Args:
            path (str): path to the dataset folder
            logs (list): list of log folders
            cameras (list): list of camera folders'
            sensors (str): name of the sensors folder
            fields (list): list of sensor fields to read
            transform (torchvision.transforms): transforms to apply to image
            seq (int): number of images in the sequence
            delta (int): distance between consecutive timeframes
            grayscale (bool): grayscale image flag.
        """

        super().__init__(path, config["logs"], config["cameras"], transform, config["no_seq"], config["delta"], config["grayscale"])

        self.fields = config["sensor_fields"]
        self.sen_filelist = []

        # TODO: Better loop
        # TODO: Better loading
        for log in self.logs:  # For each log
            for _ in self.cameras:
                path_sensors = reduce(os.path.join, [path, log, sensors])

                file_list_sensors = [
                    os.path.join(path_sensors, f)
                    for f in os.listdir(path_sensors) if f.endswith(".json")
                ]
                file_list_sensors.sort(key=lambda f: int(re.sub('\D', '', f)))

                n_camera = len(file_list_sensors)  # Number of images per camera
                points = np.arange(self.delta * (self.seq - 1),
                                   n_camera - self.delta)  # Usable images
                seqs = [
                    np.arange(
                        points[idx] - self.delta * (self.seq - 1),
                        points[idx] + self.delta + 1,
                        self.delta,
                    ).astype(int) for idx in range(len(points))
                ]  # Sensor sequences indices

                for ss in seqs:
                    sen_files = [file_list_sensors[s] for s in ss]
                    self.sen_filelist.append(sen_files)


    def __getitem__(self, index):
        """Get single image/sensor sequence. 
        
        Args:
            index (int): dataset index.
        
        Returns:
            images (torch.Tensor): tensor containing the transformed image.
            sensors (torch.Tensor): tensor containing corresponding sensor data.
            seq_img (list): paths to the image files (needed for the inference comparison).
            seq_sen (list): paths to the corresponding sensor data (needed for the inference comparison).
        """

        # TODO: Add proper normalization
        seq_img = self.seq_filelist[index]
        # images = np.array([cv2.imread(s, self.cv2_flag)
        #                    for s in seq_img]) / 255.0
        images = np.array([np.asarray(Image.open(s).convert('L'))
                    for s in seq_img]) / 255.0

        if self.grayscale:
            images = np.expand_dims(images, axis=3)
        images = torch.FloatTensor(images)
        images = images.permute(0, 3, 1, 2)

        seq_sen = self.sen_filelist[index]

        sensors = []

        for s in seq_sen:
            temp_dict = json_read(s)
            sensors.append([temp_dict[f] for f in self.fields])

        sensors = np.array(sensors)

        # sensors = np.array([json_read(s)[self.fields] for s in seq_sen])
        # sensors = np.array([list(json_read(s).values()) for s in seq_sen])

        if len(sensors.shape) == 1:
            sensors = np.expand_dims(sensors, axis=1)

        sensors = torch.FloatTensor(sensors)

        # No transform atm
        return images, sensors, seq_img, seq_sen

    def __len__(self):
        return len(self.seq_filelist)


# class TempAVLatentPredictionDataset(Dataset):
#     """Class TempAVLatentPredictionDataset.
#     Pre-processed image sequences in latent with velocity values.

#     TODO: Fix .txt
#     """
#     def __init__(self, path):
#         """
#         Args:
#             path (str): path to the datset folder
#         """
#         self.path = path  # Path where latent preprocessed data is stored
#         # self.file_list = [f for f in os.listdir(self.path) if f.endswith(".txt")]

#         self.file_list = []
#         self.files_per_log = []
#         self.indices_start = []

#         # Iterate through logs
#         for p in path:
#             file_list_log = [
#                 os.path.join(p, f) for f in os.listdir(p) if f.endswith(".txt")
#             ]

#             self.indices_start.append(len(
#                 self.file_list))  # 0, 10000, 25000, etc
#             self.files_per_log.append(len(file_list_log))  # 10000, 15000, etc
#             self.file_list += file_list_log

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, index):
#         # data = np.loadtxt(self.path+"/"+self.file_list[index])
#         data = np.loadtxt(self.file_list[index])
#         return torch.FloatTensor(data)

#     def get_fractional_log_indices(
#             self,
#             split_train=0.8,
#             split_val=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """TODO: Finish and test.
#         Instead of using pytorch dataset splits, use splits without overlapping.
#         """

#         indices_train, indices_val, indices_test = [], [], []

#         for i in len(self.path):
#             idx_end_train = int(np.floor(self.files_per_log * split_train))
#             idx_end_val = int(
#                 np.floor(self.files_per_log * (split_train + split_val)))

#             idx_train = range(0, idx_end_train) + self.indices_start[i]
#             idx_val = range(idx_end_train, idx_end_val) + self.indices_start[i]
#             idx_test = (range(idx_end_val, self.files_per_log - 1) +
#                         self.indices_start[i])

#             indices_train += idx_train
#             indices_val += idx_val
#             indices_test += idx_test

#         return indices_train, indices_val, indices_test


# class VehicleDataset(Dataset):
#     def __init__(
#         self,
#         no_seq: int,
#         delta: int,
#         path_preprocessed: str,
#         path_latent: str,
#         path_csv: str,
#     ):

#         self.no_seq = no_seq
#         self.delta = delta

#         self.preprocessed_path = path_preprocessed
#         self.latent_path = path_latent
#         self.csv_path = path_csv  # Path where csv with abs velocities is stored

#         self.file_list = [
#             f for f in os.listdir(self.latent_path) if f.endswith(".txt")
#         ]
#         self.file_list.sort()
#         self.time_stamps = np.array(
#             [os.path.splitext(x)[0] for x in self.file_list]).astype("int64")
#         self.time_stamps.sort()
#         self.usable_time_stamps = self.time_stamps[delta * (
#             self.no_seq - 1
#         ):-delta]  # delete first no_pics and last (required for lstm evaluation)

#         self.vel_data = pd.read_csv(self.csv_path + "/velocity_abs.csv")

#     def __len__(self):
#         return len(self.usable_time_stamps)

#     # def __getitem__(self, index):
#     #     sequence = self.load_sequence(indexes)
#     #     stacked_batch = torch.cat(all_seq,dim=2)
#     #     stacked_batch = stacked_batch.permute(0,2,1)
#     #     return stacked_batch

#     def __saveitem__(self, indexes):
#         """Saves augmented latent sequence to a single file

#         Args:
#             indexes (list):
#         """
#         sequence = self.load_sequence(indexes)
#         np.savetxt(
#             self.preprocessed_path + self.file_list[indexes[self.no_seq - 1]],
#             sequence.cpu().numpy(),
#         )

#     def load_sequence(self, sequence):
#         """Loads and concatenates latent and sensor data.

#         Args:
#             sequence (list): list of indexes in the sequence
#         """
#         latent = np.array([
#             np.loadtxt(
#                 os.path.join(self.latent_path, f"{self.time_stamps[idx]}.txt"))
#             for idx in sequence
#         ])
#         abs_vel = np.array([
#             self.vel_data.iloc[(
#                 self.vel_data["time_stamp"] -
#                 self.time_stamps[idx]).abs().argsort()[:1]]["abs_vel"].values
#             for idx in sequence
#         ])
#         return torch.FloatTensor(np.concatenate((latent, abs_vel), axis=1))


class SimpleMovingMNISTDataset(Dataset):
    """Simple dataset for moving mnist"""
    def __init__(self, path: str, seq: int = 5):
        """
        Args:
            path (str): path to the moving MNIST Dataset
            seq (int): number of images in the sequence
        """

        assert seq <= 19, "Maximum training sequence length is 19 (20-1)"

        self.path = path
        self.seq = seq

        # Load and normalize moving MNIST Dataset
        self.data = np.transpose(np.load(self.path), (1, 0, 2, 3)) / 255.0
        # Expand dims to imitate channels
        self.data = np.expand_dims(self.data, 2)

    def __getitem__(self, index):
        """Get single image sequence."""

        sequence = self.data[index, :self.seq + 1, :, :, :]
        return torch.FloatTensor(sequence)

    def __len__(self):
        return self.data.shape[0]


class SingleFolderDataset(Dataset):
    """No log folders, all sequences in the same folder(s)."""

    def __init__(
        self,
        path: str,
        cameras: List[str],
        transform=None,
        grayscale: bool = False,
    ):
        """
        Args:
            path (str): path to the dataset folder
            cameras (list): list of camera folders
            transform (torchvision.transforms): transforms to apply to image
        """

        self.path = path
        self.cameras = cameras
        self.grayscale = grayscale
        self.file_list = []  # Construct file list (multiple folders)
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if self.grayscale else cv2.IMREAD_COLOR

        # Compose transforms
        if transform is not None:
            self.transform = transforms.Compose([transform, transforms.ToTensor(),])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])

        # TODO: Better loop
        for camera in cameras:
            path_camera = os.path.join(path, camera)
            file_list_camera = [
                os.path.join(path_camera, f)
                for f in os.listdir(path_camera)
                if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")
            ]
            self.file_list += file_list_camera

    def __len__(self):
        """Return dataset length."""
        return len(self.file_list)

    def __getitem__(self, index):
        """Get single image."""
        image = cv2.imread(self.file_list[index], self.cv2_flag)
        return self.transform(image), self.file_list[index]

    def get_filename(index):
        return self.file_list[index]


