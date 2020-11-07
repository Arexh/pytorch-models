from data_loader.criteo_dataset import CriteoDataset
from data_loader.my_criteo_dataset import MyCriteo
from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class CriteoDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = CriteoDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MyCriteoLoader(BaseDataLoader):
    def __init__(self,
                 data_dir,
                 batch_size,
                 sparse_norm=False,
                 cache_path='cache/criteo',
                 rebuild_cache=False,
                 pin_memory=False,
                 train=True,
                 shuffle=True,
                 validation_split=0.0,
                 num_workers=1):
        self.data_dir = data_dir
        self.dataset = MyCriteo(data_dir=data_dir,
                                sparse_norm=sparse_norm,
                                cache_path=cache_path,
                                rebuild_cache=rebuild_cache,
                                train=train)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, pin_memory)
