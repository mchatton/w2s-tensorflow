from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import math
import glob
import argparse

import yaml
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from options.options import parse
from utils import train_util, util
from models.models import make_generator, make_discriminator


from data_scripts import data_loader
from data_scripts.data_loader import DatasetLoader, decode_img_from_path

def main():
    ## options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    if args.opt is not None:
        opt = parse(args.opt)
    else:
        opt = parse('./options/train_ESRGAN.yml')

    # Instantiate the generator
    model = make_generator(opt, print_summary=True)
    
    # Load the weights from a .h5 file
    model.load_weights('./experiments/backup/full/RRDB_GAN.h5')
    
    # Add a dummy Input layer that allows for inputs of arbitrary size (needed for SavedModel format)
    input = tf.keras.layers.Input(shape=(None,None,1))
    
    # Create a new model with the input layer
    out = model(input)
    newModel = tf.keras.models.Model(input,out)
    
    # Save the model with the SavedModel format
    newModel.save('./experiments/pretrained_models/RRDB_GAN', save_format='tf')
    
    
if __name__ == '__main__':
    main()
