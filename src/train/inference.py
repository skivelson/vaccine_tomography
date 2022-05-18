import os, sys
import h5py
import torch
import numpy as np
from tqdm import tqdm
import yaml

from models import UNet3D
from datasets import TomoDataset

sys.path.append('..')
import config
import arg_parser


def predict(model, dataset, file_name, output_dir):
    for i, (_, tomo) in enumerate(tqdm(dataset.data_list)):
        
        if file_name not in tomo:
            continue
        
        out_path = os.path.join(output_dir, tomo)
        data = np.expand_dims(dataset[i]["data"], 0)
        d, h, w = dataset[i]["shape"]
       
        with torch.autocast("cuda"):
            preds = model(torch.tensor(data).to("cuda")).to("cpu").numpy().squeeze()
        
        # transform back to 0-255 from 0-1
        data = (255 * data).squeeze().astype(np.uint8)
                
        with h5py.File(out_path, 'w') as fh:
            fh.create_dataset("data", data=data)
            fh.create_dataset("preds", data=preds.astype(np.float32))


def main():
    args = arg_parser.get_args(sys.argv[0])

    test_dataset = TomoDataset(os.path.join(config.TRAIN_TOMO_DIR, args.data))


    ckpt_path = config.LOG_DIR + f'/{args.exp_name}/version_{args.version}/checkpoints'
    ckpt_path += '/' + os.listdir(ckpt_path)[0]
    model = UNet3D.load_from_checkpoint(ckpt_path)
    model.eval()
    model.to("cuda")

    out_dir=config.PREDICTIONS_DIR + f'/{args.exp_name}-version_{args.version}/'
    if len(args.exp_name) == 0:
        out_dir=config.PREDICTIONS_DIR + f'/version_{args.version}/'
    
    os.makedirs(out_dir, exist_ok=True)

    with torch.inference_mode():
        predict(model, test_dataset, args.file_name, out_dir)


if __name__ == '__main__':
    main()
