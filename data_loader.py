import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from base.base_data_loader import BaseDataLoader
from base.parse_config import LoadConfig


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, trsfm):
        self.imgs = [path for path in Path(data_dir + "train").rglob("*.png")]
        self.train = pd.read_csv(os.path.join("data/", "train.csv"))
        self.trsfm = trsfm

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        id = self.imgs[idx].name.split(".")[0]  # removesuffix(".png")
        try:
            label = int(self.train[self.train["ID"] == id]["LABEL"])
        except:
            # images without labels
            label = 0
        img = self.trsfm(img)
        return img, label

    @property
    def classes(self):
        return ["negative", "positive"]

    @property
    def class_to_idx(self):
        return {c: i for i, c in enumerate(self.classes)}


class DataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
    ):
        trsfm = transforms.Compose(
            [
                transforms.RandomResizedCrop((512,)),
                # https://docs.habana.ai/en/v1.2.0/PyTorch_User_Guide/PyTorch_User_Guide.html#current-limitations
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.data_dir = data_dir
        self.dataset = Dataset(data_dir=data_dir, trsfm=trsfm)
        # import torchvision
        # self.dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=trsfm)
        if num_workers<= 2:
            num_workers = num_workers
        else:
            num_workers = torch.multiprocessing.cpu_count()
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )

    @property
    def train_loader(self):
        train = self.loaders.train
        return train[0]

    @property
    def valid_loader(self):
        valid = self.loaders.valid
        return valid[0]


if __name__ == "__main__":
    lc = LoadConfig(os.path.join("config.yaml"))
    conf = lc.parse_config()
    data_dir = conf.data_dir
    batch_size = conf.batch_size
    shuffle = conf.shuffle
    validation_split = conf.validation_split
    num_workers = conf.num_workers

    dl = DataLoader(data_dir, batch_size, shuffle, validation_split, num_workers)
    print(dl.train_loader.dataset.classes)
    print(len(dl.train_loader))
    print(len(dl.valid_loader.dataset))
