import os
import glob
import pathlib
import argparse


import tensorflow as tf
import numpy as np

from options.options import parse
from data_scripts.data_loader import decode_img_from_path, prepare_validation_dataset
from utils.util import make_validation_plot, tensor_numpy_to_img, save_img
from utils import util
from models.models import make_generator
import matplotlib.image as mpimg

def main():
    """
    Perform validation on the validation images folder using the trained network
    
    Args:
        opt: the options file

    """
    
    ## options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    if args.opt is not None:
        opt = parse(args.opt)
    else:
        opt = parse('./options/train_ESRGAN.yml')
        
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # build generator
    model = make_generator(opt, print_summary=False)

    if os.path.exists(opt['path']['trained_model_G']):
        print("Using trained model:",opt['path']['trained_model_G'])
        model.load_weights(opt['path']['trained_model_G'])
    else:
        checkpoints = sorted(glob.glob(opt['path']['checkpoint_root']+"*.h5"))
        if len(checkpoints) > 0:
            print("Using checkpoint:",checkpoints[-1])
            model.load_weights(checkpoints[-1])
        else:
            if os.path.exists(opt['path']['pretrained_model_G']):
                print("Using pre-trained model:",opt['path']['pretrained_model_G'])
                model.load_weights(opt['path']['pretrained_model_G'])
            else:
                print("No checkpoint found. Aborting")
                return
                
            
    # Load image directories
    hr_dir = opt['datasets']['validation']['dataroot_HR']
    lr_dir = opt['datasets']['validation']['dataroot_LR']
    
    lr_list = sorted(glob.glob(lr_dir+'*.png'))
    hr_list = sorted(glob.glob(hr_dir+'*.png'))
    filenames = [pathlib.Path(path).name for path in hr_list]
    file_iter = iter(filenames)

    # Load dataset
    ds = prepare_validation_dataset(opt)
    
    # Call prediction on all validation images and save outputs
    for lr in ds:
        out_batch = model(lr, training=False)
        for out in out_batch:
            filename = next(file_iter)
            out_path = opt['datasets']['validation']['results_dir']+filename
            save_img(tensor_numpy_to_img(out.numpy()), out_path)
            
    print("saved images")
    
    out_dir = opt['datasets']['validation']['results_dir']
    out_list = sorted(glob.glob(out_dir+'*.png'))
    
    # Make validation plots and save them
    for filename, lr, out, hr in zip(filenames, lr_list, out_list, hr_list):
        lr = mpimg.imread(lr)
        out = mpimg.imread(out)
        hr = mpimg.imread(hr)
        make_validation_plot(lr, out, hr, opt['datasets']['validation']['plots_dir']+filename) 
    
        
if __name__ == '__main__':
    main()