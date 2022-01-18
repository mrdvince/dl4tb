import os

from torchvision import datasets, transforms

from base import BaseDataLoader
from base.base_config import LoadConfig


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
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            root=data_dir, train=True, transform=trsfm, download=True
        )  # datasets.ImageFolder(data_dir, transform=trsfm)
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
    print(len(dl.train_loader.dataset))
    print(len(dl.valid_loader.dataset))
