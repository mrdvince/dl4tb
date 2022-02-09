import os
import zipfile
from glob import glob
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import gdown
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from skimage.io import imread as imread
from skimage.transform import resize as resize
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from utils import copy_images_to_folder, download
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class PROCData:
    def __init__(self):
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(rotate_limit=15, always_apply=True),
            ]
        )
        image_paths = glob(
            os.path.join(
                "data/pulmonary-chest-xray-abnormalities/Montgomery/MontgomerySet/CXR_png",
                "*.png",
            )
        )
        # fmt: off
        self.images_with_masks_paths = [
            (image_path,os.path.join("/".join(image_path.split("/")[:-2]),"ManualMask","leftMask", os.path.basename(image_path)),
                os.path.join("/".join(image_path.split("/")[:-2]),"ManualMask","rightMask",os.path.basename(image_path))) for image_path in image_paths
            ]
        # fmt: on
        self.OUT_DIM = (512, 512)

    def process(self):
        images, masks = [], []
        for mri, left_lung, right_lung in tqdm(
            self.images_with_masks_paths, position=0, leave=True
        ):
            images.append(self.image_from_path(mri))
            masks.append(self.mask_from_paths(left_lung, right_lung))

        transformed_images, transformed_masks = [], []

        for image, mask in zip(images, masks):
            sample = {"image": image.copy(), "mask": mask.copy()}
            out = self.transform(**sample)
            transformed_images.append(out["image"])
            transformed_masks.append(out["mask"])

        image_dataset = images.copy() + transformed_images
        mask_dataset = masks.copy() + transformed_masks

        x_train, x_val, y_train, y_val = train_test_split(
            image_dataset, mask_dataset, test_size=0.2
        )
        scaler = StandardScaler()
        x_train = scaler.fit_transform(
            np.array(x_train).reshape(-1, 512 * 512)
        ).reshape(-1, 512, 512)
        x_val = scaler.transform(np.array(x_val).reshape(-1, 512 * 512)).reshape(
            -1, 512, 512
        )
        return x_train, y_train, x_val, y_val

    def image_from_path(self, path):
        img = resize(imread(path), self.OUT_DIM, mode="constant")
        return img

    def mask_from_paths(self, path1, path2):
        img = resize(
            cv2.bitwise_or(imread(path1), imread(path2)), self.OUT_DIM, mode="constant"
        )
        return img


class UNETDataModule(pl.LightningDataModule):
    def __init__(self, config=None):
        super(UNETDataModule, self).__init__()
        self.project_root = os.getcwd() + "/"  # hydra.utils.get_original_cwd() + "/"
        print(self.project_root)

    def prepare_data(self):
        if not os.path.exists(
            self.project_root + "data/" + "pulmonary-chest-xray-abnormalities"
        ):
            download(
                "https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities",
                self.project_root + "data",
            )

    def setup(self, stage: Optional[str] = None):
        x_train, y_train, x_val, y_val = PROCData().process()
        self.train_ds = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        self.val_ds = TensorDataset(torch.tensor(x_val), torch.tensor(y_val))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=32, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=32, shuffle=False)


class ClassifierDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super(ClassifierDataModule, self).__init__()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.config = config
        self.project_root = hydra.utils.get_original_cwd() + "/"
        self.data_dir = self.project_root + self.config.processing.data_dir + "/"

    def prepare_data(self) -> None:
        filename = self.config.processing.zip_file
        path = Path(self.project_root + "data/")
        path.mkdir(parents=True, exist_ok=True)
        if not path.joinpath(filename).exists():
            gdown.download(
                self.config.processing.data_url,
                path.joinpath(filename).as_posix(),
                quiet=False,
            )
            # extract the zip file
            with zipfile.ZipFile(str(path / filename), "r") as zip_ref:
                zip_ref.extractall(path / filename.replace(".zip", ""))
            copy_images_to_folder("data/tb_data/train")

    def setup(self, stage: Optional[str] = None) -> None:
        # load data
        if stage in (None, "fit"):
            self.train_dataset = datasets.ImageFolder(
                self.data_dir + "proc_tb", self.transforms
            )

            train_samples = int(len(self.train_dataset) * 0.8)

            self.train_data, self.val_data = random_split(
                self.train_dataset,
                [train_samples, len(self.train_dataset) - train_samples],
            )
        if stage in (None, "test"):
            self.test_data = datasets.ImageFolder(
                self.project_root + self.data_dir + "/test/",
                transform=transforms.ToTensor(),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=32,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=32,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=32, num_workers=6)


if __name__ == "__main__":
    train_loader = UNETDataModule()
    train_loader.prepare_data()
    train_loader.setup()
    train_loader = train_loader.train_dataloader()
    data = next(iter(train_loader))
    loader_images, loader_masks = data
    print(loader_images.shape, loader_masks.shape)
