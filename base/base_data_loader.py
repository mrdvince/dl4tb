from collections import namedtuple

import numpy as np
from torch.utils import data


class BaseDataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        validation_split,
        num_workers,
        collate_fn=data.dataloader.default_collate,
    ):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.n_samples = len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_sampler, self.valid_sampler = self._split_sampler(
            self.validation_split
        )

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }

    def _split_sampler(self, v_split):
        idx_full = np.arange(self.n_samples)
        np.random.seed(69420)
        np.random.shuffle(idx_full)
        len_valid = int(self.n_samples * v_split)
        train_idx, valid_idx = idx_full[len_valid:], idx_full[:len_valid]
        train_sampler = data.sampler.SubsetRandomSampler(train_idx)
        valid_sampler = data.sampler.SubsetRandomSampler(valid_idx)

        return train_sampler, valid_sampler

    # using experimental dataloader
    @property
    def loaders(self):
        loaders = namedtuple("loaders", ["train", "valid"])
        train = (data.DataLoader(**self.init_kwargs, sampler=self.train_sampler),)

        valid = (data.DataLoader(**self.init_kwargs, sampler=self.valid_sampler),)
        return loaders(train, valid)
