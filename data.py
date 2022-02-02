import os
import zipfile
from pathlib import Path
from typing import Optional

import gdown
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, trsfm):
        self.imgs = [path for path in Path(data_dir).rglob("*.png")]
        self.train = pd.read_csv(os.path.join("data/", "train.csv"))
        self.trsfm = trsfm

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        id = self.imgs[idx].name.split(".")[0]  # removesuffix(".png")
        label = int(self.train[self.train["ID"] == id]["LABEL"])
        img = self.trsfm(img)
        return img, label


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/home/vinc3/Projects/dl4tb/data/tb_data"):
        super(DataModule, self).__init__()
        self.transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.data_dir = data_dir

    def prepare_data(self) -> None:
        # download data
        path = "data"
        filename = "tb_data.zip"
        url = "https://drive.google.com/uc?id=1KdpV3M27kV-_QOQOrAentfzZ2tew8YS-&"
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not path.joinpath(filename).exists():
            gdown.download(url, path.joinpath(filename).as_posix(), quiet=False)
            # extract the zip file
            with zipfile.ZipFile(str(path / filename), "r") as zip_ref:
                zip_ref.extractall(path / filename.replace(".zip", ""))

    def setup(self, stage: Optional[str] = None) -> None:
        # load data
        if stage in (None, "fit"):
            self.train_dataset = Dataset(
                data_dir=f"{self.data_dir}/train/", trsfm=self.transforms
            )

            train_samples = int(len(self.train_dataset) * 0.8)

            self.train_data, self.val_data = random_split(
                self.train_dataset,
                [train_samples, len(self.train_dataset) - train_samples],
            )
        if stage in (None, "test"):
            self.test_data = Dataset(
                data_dir=f"{self.data_dir}/test/", trsfm=transforms.ToTensor()
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=32,
            shuffle=True,
            num_workers=6,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=32,
            shuffle=False,
            num_workers=6,
        )

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=32, num_workers=6)
