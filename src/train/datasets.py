import os
import h5py
import math
import numpy as np
from torch.utils.data import Dataset



class TomoDataset(Dataset):
    def __init__(self, data_dir, filter_empty=False):
        self.filter_empty = filter_empty
        self.data_list = [(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".hdf")]
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        directory, file_name = self.data_list[idx]
        path = os.path.join(directory, file_name)
        
        with h5py.File(path) as fh:
            # this is the input volume
            data = fh["data"][()].astype(np.float32)
            assert data.shape[-1] <= 512
            # these are the labels we use for training
            labels = fh["labels"][()].astype(np.float32)

            # optionally skips tomograms with no labels
            if self.filter_empty and labels.max() < 1.0:
                return self[(idx + 1) % len(self)]

            
            
        d, h, w = data.shape
        # we normalize the data - can probably do better than minmax
        data = (data - data.min()) / np.ptp(data)
        # per pixel loss weights - 0 for unlabelled regions
        weight = np.where(labels < 0, 0.0, 1.0).astype(np.float32)
        
        return {
            "data": np.expand_dims(self.pad(data), 0),
            "labels": np.expand_dims(self.pad(labels), 0),
            "weight": self.pad(weight).astype(bool),
            "shape": data.shape,
        }

    def pad(self, x, r=32):
        d, h, w = x.shape

        d_new = math.ceil(d / r) * r
        h_new = math.ceil(h / r) * r
        w_new = math.ceil(w / r) * r

        if (d_new, h_new, w_new) == x.shape:
            return x

        x_new = np.zeros((d_new, h_new, w_new), dtype=x.dtype)
        x_new[:d, :h, :w] = x
        return x_new


if __name__ == '__main__':
    pass 
