import argparse



def get_args(file_name):
        file_name = file_name.split('/')[-1]
        parser = argparse.ArgumentParser()

        if file_name == 'get_slices.py':
                parser.add_argument("--data", type=str, 
                        help='data set name')

    
        if file_name == 'generate_train_data.py':
                parser.add_argument("--data", type=str, 
                        help='data set name')
                parser.add_argument("--crop", type=int, default=-1,
                        help='crop size. default value indicates no crop')
                parser.add_argument("--crop_only", action='store_true', default=False,
                        help='if False, crop from existing data in train_tomos')
                parser.add_argument('--skip_tomos', nargs="+", default=[],
                        help='if a tomogram name has any of the given args as a substring, skip it')
        
        if file_name == 'train.py':
                parser.add_argument("--data", type=str,
                        help='name of train dataset in train_tomos to use')
                parser.add_argument("--exp_name", type=str, default='default',
                        help='experiment ame to save in logs')
                parser.add_argument("--channels", nargs='+', default=[8, 16, 32, 64],
                        help='layer sizes for unet')
        
        if file_name == 'inference.py':
                parser.add_argument('--exp_name', type=str, default='',
                        help='name of folder in log with experiment')
                parser.add_argument('--version', type=int, default=0,
                        help='version in exp_name to open')
                parser.add_argument('--data', type=str, default='',
                        help='folder with data in it')
                parser.add_argument('--file_name', type=str, default='',
                        help='optional to provide a name of a file or substring of file name \
                                only files with this substring present will be loaded')


        return parser.parse_args()

