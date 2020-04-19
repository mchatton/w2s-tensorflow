import os
import glob
import pathlib

import cv2
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from options.options import parse
    
def decode_img_from_path(file_path):
    """
    Loads a grayscale image from its path and returns it
    
    Args:
        file_path: the path of the image
        
    Returns:
        The requested image
        
    """
    
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img,channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    return tf.expand_dims(img,0)

def main():

    # Load the pretrained model
    model = tf.keras.models.load_model('experiments/pretrained_models/RRDB_GAN',compile=False)
    
    #Load the iamge directory
    input_dir = 'experiments/input/'
    input_list = sorted(glob.glob(input_dir+'*.png'))
    filenames = [pathlib.Path(path).name for path in input_list]
    file_iter = iter(filenames)


    n_images = len(input_list)
    i = 0
    # For each image in input directory
    for input in input_list:
    
        # Decode the image into a tensor
        img = decode_img_from_path(input)
        
        # Apply super-resolution and denoising
        output = model(img).numpy()[0]
        
        # Convert the image into a savable 8-bit image
        output = (255.0 / output.max() * (output - output.min())).astype(np.uint8)
        
        # Save the image to the output directory
        filename = pathlib.Path(input).name
        out_path = 'experiments/output/'+filename
        cv2.imwrite(out_path,output)
        i += 1
        print("Saved "+str(i)+"/"+str(n_images))
            
    print("DONE")
        
if __name__=='__main__':

    main()