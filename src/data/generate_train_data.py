
"""Generate Train Data
Reads the annotations downloaded from hasty.ai and makes labels
The mito and granule labels are stored along with the raw_data
Z limits from the annotation file are used to annotate slices
which are confirmed to have no mito or granules

"""
import os
import sys

import h5py
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.image as mpimg
import torch
from PIL import Image 

sys.path.append("..")
import config
import arg_parser


def skip(file_name, skip_list):
    for key in skip_list:
        if key in file_name:
            print('~~~~~ skipping', file_name, '~~~~~')
            return True 
    return False 

def get_train_tomo_from_label_slices(df, data_set, skip_list):

    dst_dir = os.path.join(config.TRAIN_TOMO_DIR, data_set) 
    os.makedirs(dst_dir, exist_ok=True)
    avg_pool = torch.nn.AvgPool2d(2, stride=2)
    max_pool = torch.nn.MaxPool2d(2, stride=2)

    for _, row in df.iterrows():
        z_min, z_max = row['z_min'], row['z_max']

        if skip(row['tomo_name'], skip_list):
            continue 

        input_tomo_path = os.path.join(config.RAW_TOMO_DIR, data_set, row['tomo_name'])
        output_tomo_path = os.path.join(config.TRAIN_TOMO_DIR, data_set, row['tomo_name'].replace('preproc', 'postproc'))


        with h5py.File(input_tomo_path) as fh:
            data = fh['MDF']['images']['0']['image'][()]
            assert data.shape == (400, 1024, 1024)
        
        vax_labels = -1 * np.ones_like(data, dtype=np.int8)
        vax_labels[:z_min] = 0
        vax_labels[z_max:] = 0

        tomo_prefix = row['tomo_name'].split('__')[0]
        for labeled_slice in os.listdir(os.path.join(config.VAX_ANNOTATIONS, data_set)):
            if '._' in labeled_slice or tomo_prefix not in labeled_slice:
                continue
            slice_path = os.path.join(config.VAX_ANNOTATIONS, data_set, labeled_slice)
            # print(labeled_slice)
            # print(slice_path)
            # exit()
            slice_number = int(slice_path.split('_')[-1].replace('.png', ''))
            label = np.asarray(Image.open(slice_path))
            vax_labels[slice_number] = np.where(label > 0, 1, 0).astype(np.int8)
        # a, b = torch.tensor(data), torch.tensor(vax_labels.astype(np.float32))
        # print('\n', '='*10, a.type(), b.type())
        # print('\n', '='*10, a.dtype, b.dtype)
        # data = avg_pool(torch.tensor(data)).numpy()
        # print('\n', '='*10, data.type(), data.dtype)
        vax_labels = max_pool(torch.tensor(vax_labels.astype(np.float32))).numpy().astype(np.int8)
        data = avg_pool(torch.tensor(data)).numpy()

        with h5py.File(output_tomo_path, 'w') as fh:
            fh.create_dataset("data", data=data, compression="gzip")
            fh.create_dataset("labels", data=vax_labels, compression="gzip")

class fix_depth():
    def __init__(self,  depth=192):
        self.depth = depth 
        
    def __call__(self, x, z_min, z_max):
        start = max(0, z_min - (200 - (z_max - z_min)) // 2)
        assert (start + 200) >= z_max 
        return x[start:start+self.depth]


class GridDivide3D():
    def __init__(self, dims=64):
        self.d = 0
        self.w = 0
        self.h = 0
        if type(dims) == int:
            self.d, self.w, self.h = dims, dims, dims
        else:
            self.d, self.w, self.h = dims
    
    def __call__(self, input_vol):
        assert len(input_vol.shape) == 3
        output_vols = []
        x, y, z = input_vol.shape
        for i in range(x//self.d):
            for j in range(y//self.w):
                for k in range(z//self.h):
                    vol = input_vol[i*self.d:(i+1)*self.d, j*self.w:(j+1)*self.w, k*self.h:(k+1)*self.h]
                    title = (f'{i}-{j}-{k}')
                    output_vols.append({'vol' : vol, 'title' : title})
        return output_vols

def get_cropped_vols_from_train_tomos(df, data_set, skip_list, dims=64):
    
    depth_transform = fix_depth(min(dims*3, 300))
    crop_transform = GridDivide3D(dims)

    for _, row in df.iterrows():
        z_min, z_max = row['z_min'], row['z_max']
        tomo_name = row['tomo_name'].replace('preproc', 'postproc')

        if skip(tomo_name, skip_list):
            continue 

        if tomo_name not in os.listdir(os.path.join(config.TRAIN_TOMO_DIR, data_set)):
            print(f'Skipping {tomo_name} - not in train_tomos/{data_set}')
            continue

        input_tomo_path = os.path.join(config.TRAIN_TOMO_DIR, data_set, tomo_name)
        if not os.path.exists(input_tomo_path):
            raise RuntimeError(f"{data_set} tomos need initial proceessing before cropping. Try running \n python generate_train_data.py --data {data_set} --crop {dims}")

        dst_dir = os.path.join(config.TRAIN_TOMO_DIR, data_set + f'_{dims}')
        os.makedirs(dst_dir, exist_ok=True)

        with h5py.File(input_tomo_path, 'r') as fh1:
            data = fh1["data"][()]
            labels = fh1["labels"][()]
            assert data.shape == labels.shape
            assert data.shape == (400, 512, 512)  
        
            data = depth_transform(data, z_min, z_max)
            labels = depth_transform(labels, z_min, z_max)
            assert (data.shape[1],data.shape[2]) == (512, 512)

        dvols = crop_transform(data)
        lvols = crop_transform(labels)

        for i in range(len(data)):
            if lvols[i]['vol'].max() < 1:
                continue 
            out_path = os.path.join(dst_dir, tomo_name.replace('.', f'_{dvols[i]["title"]}.'))
            with h5py.File(out_path, 'w') as fh2:
                fh2.create_dataset("data", data=dvols[i]['vol'], compression="gzip")
                fh2.create_dataset("labels", data=lvols[i]['vol'], compression="gzip")


def main():
    args = arg_parser.get_args(sys.argv[0])
    print(args.crop_only)

    annotation_file = os.path.join(config.ANNOTATION_CSV, f"{args.data}.csv")
    
    if not os.path.exists(annotation_file):
        print('==>', annotation_file)
        raise RuntimeError(f"No annotations for {args.data} were found")
    df = pd.read_csv(annotation_file)
    
    if not args.crop_only:
        print('\nCreating train tomos from raw tomos ....\n')
        get_train_tomo_from_label_slices(df, args.data, args.skip_tomos)  
        
    if args.crop > -1:
        print('\nCreating crop vols from train tomos ....\n')
        get_cropped_vols_from_train_tomos(df, args.data, args.skip_tomos, args.crop)
        

        
if __name__ == '__main__':
    main()

