import tifffile
from torch.utils.data import Dataset
import numpy as np


class ISBI2012Dataset(Dataset):
    def __init__(self, path_img, path_target, transforms=None):
        self.train = np.expand_dims(tifffile.TiffFile(path_img).asarray(), axis=-1)
        self.targets = np.expand_dims(tifffile.TiffFile(path_target).asarray(), axis=-1)
        self.transforms = transforms

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        img = self.train[idx]
        target = self.targets[idx]

        if self.transforms:
            img = self.transforms(img)
            target = self.transforms(target)

        return img, target
