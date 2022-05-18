import os
import sys

import h5py
import pandas as pd
from skimage import io

sys.path.append("..")
import config
import arg_parser


def main():
    args = arg_parser.get_args(sys.argv[0])
    csv_path = os.path.join(config.ANNOTATION_CSV, f"{args.data}.csv")
    dst_dir = os.path.join(config.SLICES_TO_LABEL, args.data)
    
    if not os.path.exists(csv_path):
        raise RuntimeError(f"No annotation csv for {args.data} were found")
    
    os.makedirs(dst_dir, exist_ok=True)
    df = pd.read_csv(csv_path).drop(columns=["z_min", "z_max"])
    
    for _, row in df.itertuples():
        tomo_name = row['tomo_name']
        tomo_path = os.path.join(config.RAW_TOMO_DIR, args.data, tomo_name)
        
        with h5py.File(tomo_path) as fh:
            for i in range(5):
                s = row[f'slice_{i}']
                out_path = os.path.join(dst_dir, tomo_name.replace('preproc.hdf', f'{s}.png'))
                io.imsave(out_path, fh['MDF']['images']['0']['image'][s])


if __name__ == '__main__':
    main()