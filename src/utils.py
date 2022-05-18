import os, sys
import numpy as np
import h5py 
import torch 
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import pandas as pd 
import config


def plot_slices(slices):
    fig, axs = plt.subplots(1, len(slices))
    fig.set_size_inches(5*len(slices), 5)
    for i, (k, s) in enumerate(slices):
        axs[i].imshow(s, interpolation='nearest')
        axs[i].set_xlabel(k)
    plt.show()

def inspect_label(path, s):
    slices = []
    with h5py.File(path, 'r') as fh:
        for k in fh.keys():
            slices.append((k, fh[k][()][s]))
        # data = fh['data'][()]
        # labels = fh['labels'][()]
    #print(f'Label contains ', np.unique(labels))
    # plot_slices([data[s], labels[s]])
    plot_slices(slices)

def plot_metrics(partial_path, plot=['dice_loss','bce_loss']):
    path = os.path.join(config.LOG_DIR, partial_path, 'metrics.csv')
    df = pd.read_csv(path)
    df[plot].plot()
    return df