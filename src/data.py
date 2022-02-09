import os
import zipfile
from glob import glob
from pathlib import Path
from typing import Optional

import albumentations as A
import gdown
import hydra
import numpy as np
import pytorch_lightning as pl
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utils import copy_images_to_folder, download


class UNETDataset:
    def __init__(self, cxr_dir, mask_dir, transform=None):
        self.cxr_images = glob(os.path.join(cxr_dir, "*.png"))
        self.mask_images = glob(os.path.join(mask_dir, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.cxr_images)

    def __getitem__(self, idx):
        cxr_png_path = Path(self.cxr_images[idx])
        mask_png_path = Path(self.mask_images[idx])
        img = np.array(Image.open(cxr_png_path).convert("RGB"))
        mask = np.array(Image.open(mask_png_path).convert("L"), dtype=np.float32)
        mask = mask / 255

        if self.transform:
            augs = self.transform(image=img, mask=mask)
            img = augs["image"]
            mask = augs["mask"]

        return img, mask


class UNETDataModule(pl.LightningDataModule):
    def __init__(self, config=None):
        super(UNETDataModule, self).__init__()
        self.project_root = os.getcwd() + "/"  # hydra.utils.get_original_cwd() + "/"
        dims = (256, 256)
        self.transforms = A.Compose(
            [
                A.Resize(height=dims[0], width=dims[1], always_apply=True),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    def prepare_data(self):
        if not os.path.exists(
            self.project_root + "data/" + "pulmonary-chest-xray-abnormalities"
        ):
            download(
                "https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities",
                self.project_root + "data",
            )

    def setup(self, stage=None):
        dataset = UNETDataset(
            cxr_dir=self.project_root + "data/proc_seg/cxr_pngs",
            mask_dir=self.project_root + "data/proc_seg/mask_pngs",
            transform=self.transforms,
        )
        train_samples = int(len(dataset) * 0.8)
        self.train_data, self.val_data = random_split(
            dataset,
            [train_samples, len(dataset) - train_samples],
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=32, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=32, shuffle=False)


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
