import os
import sys
from collections import OrderedDict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage import measure, metrics

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

#########
# plots #
#########
def make_validation_plot(lr, output, hr, save_path):
    lr = image_float_to_int(lr).squeeze()
    output = image_float_to_int(output).squeeze()
    hr = image_float_to_int(hr).squeeze()
    psnr = metrics.peak_signal_noise_ratio(hr, output)
    ssim = metrics.structural_similarity(hr, output)
    fig, ax = plt.subplots(1, 3, figsize=(20,5))
    ax[0].imshow(lr)
    ax[0].set_title('low-res image')
    ax[1].imshow(output)
    ax[1].set_title('output image: psnr={0:.3f}, ssim={1:.3f}'.format(psnr,ssim))
    ax[2].imshow(hr)
    ax[2].set_title('ground truth image')
    plt.tight_layout()
    fig.savefig(save_path)
    print("save to:",save_path)
    plt.close()
        
####################
# image convertion #
####################

def image_float_to_int(img):
    return (255.0 / img.max() * (img - img.min())).astype(np.uint8)
    
def tensor_numpy_to_img(img):
    img = img[...,::-1]
    return image_float_to_int(img)
    


def save_img(img, img_path):
    cv2.imwrite(img_path, img)
    

#################
# miscellaneous #
#################

def remove_if_present(path):
    if  os.path.exists(path):
        os.remove(path)
