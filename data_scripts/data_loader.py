import numpy as np
from numpy import random

import os
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

def decode_img_from_path(file_path, opt, expand_dim=True):
    """
    Read an image from its path and return it
    
    Args:
        file_path the path of the image
        opt: the config file
        expand_dim whether to add a "batch" (n) dimension
        
    Returns:
        The requested image
        
    """
    
    if opt is None:
        channels = 3
    else:
        channels = opt['datasets']['train']['color_channels']
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img,channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if expand_dim:
        return tf.expand_dims(img,0)
    else:
        return img
    
        
def augment(lr_path, hr_path, opt, hflip=True, vflip=True, rot=True):
    """
    Augments an image by applying at random: horizontal/vertical flips and 90° rotations
    
    Args:
        lr_path: the path to the low resolution image
        hr_path: the path to the high resolution image
        opt: the config file
        hflip: whether to apply horizontal flip transform
        hflip: whether to apply vertical flip transform
        rot: whether to apply rotation transform
    Returns:
        The augmented image
    """
    lr = lr_path#decode_img_from_path(lr_path, opt)
    hr = hr_path#decode_img_from_path(hr_path, opt)
    if random.random() < 0.5:
    
        if hflip and random.random()<0.5:
            lr = tf.image.flip_left_right(lr)
            hr = tf.image.flip_left_right(hr)
            
        if vflip and random.random()<0.5:
            lr = tf.image.flip_up_down(lr)
            hr = tf.image.flip_up_down(hr)
            
        if rot and random.random()<0.5:
            lr = tf.image.rot90(lr)
            hr = tf.image.rot90(hr)
            
    return (lr,hr)
    
def prepare_validation_dataset(opt):
    """
    Prepare the validation dataset by loading the images from the validation folder and builing a dataset
    
    Args:
        opt: the config file
        
    Returns:
        The validation dataset
        
    """
    lr_dir = opt['datasets']['validation']['dataroot_LR']
    ds = tf.data.Dataset.list_files(lr_dir+'*.png', shuffle=False)
    
    ds = ds.map(lambda img_path: decode_img_from_path(img_path, opt, expand_dim=False), num_parallel_calls = AUTOTUNE)
    ds = ds.batch(opt['datasets']['train']['batch_size'])
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
    
def prepare_inference_dataset(opt):
    """
    Prepare the inference dataset by loading the images from the inference folder and builing a dataset
    
    Args:
        opt: the config file
        
    Returns:
        The inference dataset
        
    """
    input_dir = opt['path']['inference_root']
    ds = tf.data.Dataset.list_files(input_dir+'*.png', shuffle=False)
    
    ds = ds.map(lambda img_path: decode_img_from_path(img_path, opt, expand_dim=False), num_parallel_calls = AUTOTUNE)
    ds = ds.batch(opt['datasets']['train']['batch_size'])
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
    
    
def make_patches(lr_path, hr_path, opt):
    """
    Extract patches from a pair of hr/lr images and zip it together in a TensorFlow Dataset

    Args:
        lr_path: the path of the low-resolution image
        hr_path: the path of the high-resolution image
        opt: the config file
        
    Returns:
        A dataset of patches from the images pair

    """
    n_channels = int(opt['datasets']['train']['color_channels'])
    
    lr_size = int(opt['datasets']['train']['LR_patch_size'])
    lr_stride = int(opt['datasets']['train']['LR_stride'])
    lr = decode_img_from_path(lr_path, opt)
    
    
    hr_size = int(opt['datasets']['train']['HR_patch_size'])
    hr_stride = int(opt['datasets']['train']['HR_stride'])
    hr = decode_img_from_path(hr_path, opt)
    
    lr = tf.image.extract_patches(lr, [1,lr_size,lr_size,1], [1,lr_stride,lr_stride,1], [1,1,1,1], 'VALID')
    lr = tf.reshape(lr, [-1,lr_size,lr_size,n_channels])
    
    hr = tf.image.extract_patches(hr, [1,hr_size,hr_size,1], [1,hr_stride,hr_stride,1], [1,1,1,1], 'VALID')
    hr = tf.reshape(hr, [-1,hr_size,hr_size,n_channels])
    
    return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(lr),tf.data.Dataset.from_tensor_slices(hr)))
    
class DatasetLoader:
    """
    Class for loading the dataset
    """
    def __init__(self, opt):
        """
        Initializes the dataset loader
        
        Attributes:
            opt: the config file
        
        """
        self.batch_size = opt['datasets']['train']['batch_size']
        self.data_size = int(opt['datasets']['train']['n_images']*(opt['datasets']['train']['LR_size']/opt['datasets']['train']['LR_stride']-1)**2)
        HR_dir = opt['datasets']['train']['dataroot_HR']
        LR_dir = opt['datasets']['train']['dataroot_LR']
        #self.data_size = len(os.listdir(HR_dir))
    
        hr = tf.data.Dataset.list_files(str(HR_dir+'*.png'),shuffle=False)
        lr = tf.data.Dataset.list_files(str(LR_dir+'*.png'),shuffle=False)
        
        # Zip the two datasets together
        self.zipped_ds = tf.data.Dataset.zip((lr,hr))
    
        # Extract patches from the images (identical for lr/hr)
        self.patched_ds = self.zipped_ds.flat_map(lambda lr, hr : make_patches(lr, hr, opt))
        
    def prepare(self, ds, opt, cache_loc='datasets/cache.tfcache'):
        """
        Process the dataset to prepare it for feeding a network.
        This function first shuffle the dataset, builds minibatches and prefetch it to the device
    
        Args:
            ds: the dataset to process
            opt: the config file
            cache_loc: the path to save a cache file, used if the dataset is too large to fit in memory
        
        Returns:
            The dataset in form of a tensorflow prefetched dataset
    
        """
        
        ds = ds.cache(cache_loc)
        
        # Shuffle the dataset
        ds = ds.shuffle(self.data_size)
        
        # Apply the augmentation to each patch
        ds = ds.map(lambda lr, hr: augment(lr,hr, opt), num_parallel_calls = AUTOTUNE) # augment each patch
        
        # Build the minibatches
        ds = ds.batch(self.batch_size, drop_remainder=True)
        
        # Prefetch the dataset for faster loading
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def next(self, opt):
        """
        Get the batches for the next epoch
        
        Args:
            opt: the config file
            
        Returns:
            The dataset in form of a tensorflow prefetched dataset
    
        """
        return self.prepare(self.patched_ds, opt)