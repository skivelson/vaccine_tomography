# 3D Image Segmentation of COVID-19 Vaccine Tomograms


## Data Handling 

(Instructions for using code in vax/src/data)

Given dataset of raw tomograms "data_name", start by placing the unprocessed tomograms in 
config.RAW_TOMO_DIR/"data_name

### Annotation Steps
 
1. Use make_slices_csv.ipynb to specify slices from raw data to be label and generate 
   csv with that information. Save the information to config.ANNOTATION_CSV/data_name.csv

2. Use get_slices.py to extract the slices from the raw data and place them in 
   config.SLICES_TO_LABEL/data_name. 
   Usage:
            
         python get_slices.py --data data_name 
    
3. Once labeled, put the slice labels (masks) in config.VAX_ANNOTATIONS/data_name

### Generate Training Data from Labeled Slices 

1. Run generate_train_data.py to create train volumes by inserting labeled slices into a label volume and processing zlimits from annotation csv. These volumes will be saved to config.TRAIN_TOMOS/data_name.
  Usage: 
  
        python generate_train_data.py --date data_name 

Optional:
if you also want to create smaller subvolumes optionally run

          python generate_train_data.py --date data_name --crop 64

this will additionally create the folder config.TRAIN_TOMOS/data_name_64 in which each volume is a 64x64x64 subvolume of an element of config.TRAIN_TOMOS/data_name

if config.TRAIN_TOMOS/data_name is already populated and you only want to create config.TRAIN_TOMOS/data_name_64, run

          python generate_train_data.py --date data_name --crop 64 --crop_only

(if --crop is a number other than 64, that will be the sub vols dims and filename will be changed accordingly)

(note that before cropping, the tomo volumes will depth limited by about half, taking slices centered on the labeled slices) 


## Training and Testing Models

(Instructions for using code in vax/src/train)  

### Train 3D Unet

1. Use:
        python train.py --data data_name --exp_name exp1 
    
This will train a 3D Unet on data in the folder config.TRAIN_TOMOS/data_name. It will save the outputs to config.LOGS_DIR/exp1/version_n where n refers to the nth expeiment run under the name exp1. The default behavior if no exp name is given is to use the name "default." 

Optionally you can include the flag --channels followed by n integer values (e.g. the default behvior is equivalent to --channels 8 16 32 64). This will dictate the size of the Unet (e.g. here we would have 3 analysis layers, a bottom layer with in and out channels (32, 64), then 3 synthesis layers followed by an output layer with in and out channels (8, 1)). 

All of the argument values used in the training call will be in the same folder as theee checkpoints in the hparams.ymal file

### Run Inference on 3D Unet 

1. Use:

        python inference.py --exp_name exp1 --version n --data data_name 
        
where n is integer. 
    
This will load the tained model from the specified folder (see saving scheme from training) and make predictions on the data in config.TRAIN_TOMOS/data_name. It will save these predictions to config.PREDICTIONS_DIR/exp1-version_n.
(Note --data is the only required parameter)
     

Optionally you can include the parameter --file_name to filter which files in data_name you run inference on. For example if you run 
        
        python inference.py --exp_name exp1 --version n --data data_name --file_name bin_04

The model will only run inference on a file in data_name if bin_04 is a substring of the file's name. 

## Other

Use inspect_results notebook to looks at slices and losses using functions in vax/src/utils.py 
