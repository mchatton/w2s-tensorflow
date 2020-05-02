## Miscroscopy Image Restoration using Deep learning on W2S
by Martin Chatton, part of a Master semester project at [IVRL](https://www.epfl.ch/labs/ivrl/), EPFL, Switzerland, supervised by Ruofan Zhou and [Majed El Helou](https://majedelhelou.github.io/).


This repository provides an inference interface for joint denoising and super-resolution on grayscale microscopy images, as well as the codes samples used for training. To view and download the dataset, please refer to 

[**W2S: A Joint Denoising and Super-Resolution Dataset**](https://github.com/widefield2sim/w2s).

### [Full Report](https://arxiv.org/pdf/2004.10884.pdf)
> **Abstact:** *We leverage deep learning techniques to jointly denoise and super-resolve biomedical images acquired with fluorescence microscopy. We develop a deep learning algorithm based on the networks and method described in the recent W2S paper to solve a joint denoising and super-resolution problem. Specifically, we address the restoration of SIM images from widefield images. Our TensorFlow model is trained on the W2S dataset of cell images. On test images, the model shows a visually-convincing denoising and increases the resolution by a factor of two compared to the input image. For a 512 x 512 image, the inference takes less than 1 second on a Titan X GPU and about 15 seconds on a common CPU. We further present the results of different variations of losses used in training.*

## Inference

### Dependencies
- Python 3
- [tensorflow >= 2.1.0](https://tensorflow.org)
- numpy >= 1.17.2
- opencv-python >= 4.1.2.30 

### How to use

1. Place your widefield low-resolution images in PNG format in the `./experiments/input` folder.
2. Execute the `infer.py` script at repository root (`python3 infer.py`).
3. The script will automatically apply denoising and 2x super-resolution upscaling on all images in the input folder and save the results in the `./experiments/output` folder. You can expect a running time of 2 seconds per image if TensorFlow runs on a GPU and 15-30 seconds per image on a CPU for a ~500x500 pixels image.

### How it works

The inference process uses a model (located at `./experiments/pretrained_models/RRDB_GAN/saved_model.pb`) that was trained on 240 image pairs from the [W2S](https://github.com/widefield2sim/w2s) dataset to reproduce the resolution achieved with a SIM pipeline from a single widefield acquisition. During inference, this model is loaded and processes the images from the `./experiments/input` folder.

## Training

If you want to train your own network on your images, this section explains how to use this repository to do so.
Note: The training process is very computationally expensive. Please make sure you have GPUS with sufficient VRAM. If not, try reducing the size of the generator (`network_G:nb` field in the options file), the batch size (`train:batch_size` field in the options file) or the patch size (`train:HR_patch_size` and `train:LR_patch_size` fields in the options file).

### Dependencies
- Python 3
- [tensorflow >= 2.1.0](https://tensorflow.org)
- numpy >= 1.17.2
- opencv-python >= 4.1.2.30
- matplotlib >= 3.1.1 (for validation plots)
- PyYAML >= 5.3 (for loading config file)
- scikit-image >= 0.16.2 (for PSNR/SSIM computation used in validation plots)

### How to use

#### Training

1. Open the `./options/train_ESRGAN.yml` config file and fill the fields with your training configuration, or create your own.
2. Place your low-resolution images in PNG format in the folder specified by `train:dataroot_LR` in the config file and the corresponding high-resolution images in PNG format in the folder specified in `train:dataroot_HR`. The image pairs need to have the same names to be matched correctly, and need to be of square shape.
3. Execute the `pretrain.py` script at repository root (`python3 pretrain.py -opt path_to_option_file.yml`).
4. The pretraining will output a pretrained model in the `.h5` format at the location specified with the name and location specified in the `pretrained_model_G` field.
3. Execute the `train.py` script at repository root (`python3 train.py -opt path_to_option_file.yml`).
4. The pretraining will load the model from pretraining and output a trained model in the `.h5` format at the location specified with the name and location specified in the `trained_model_G` field.
5. If you want the model to be saved in the Tensorflow SavedModel `.pb` format, modify the `h5topn.py` file with the model you want to convert and execute it.

#### Validation

1. Place your low-resolution validation images in the folder specified by `validation:dataroot_LR` in the config file and the corresponding high-resolution images in the folder specified in `validation:dataroot_HR`. The image pairs need to have the same names to be matched correctly.
2. Execute the `validate.py` script at repository root (`python3 validate.py -opt path_to_option_file.yml`).
3. The script will automatically apply denoising and 2x super-resolution on all images in the validation LR folder and save the result images in the folder specified by `validation:results_dir`, and corresponding validation plots with PSNR and SSIM values in the folder specified by `validation:plots_dir`. You can expect a running time of 2 seconds per image if TensorFlow runs on a GPU and 15-30 seconds per image on a CPU for a ~500x500 pixels image.

### Code structure

```
.
├── README.md
├── __init__.py
├── data_scripts
│   └── data_loader.py          Script for data loading, as building patches and create Dataset objects
├── datasets
│   ├── full
│   │   ├── HR                  Default path for HR training images
│   │   └── LR                  Default path for LR training images
│   └── validation
│       └── full
│           ├── HR              Default path for HR validation images
│           ├── LR              Default path for LR validation images
│           ├── plots           Default path for validation plots
│           └── results         Default path for validation outputs
├── experiments
│   ├── input                   Default path for inference inputs
│   ├── output                  Default path for inference inputs
│   └── pretrained_models       Default path for trained models
├── h5topb.py                   Script for converting .h5 models into the .pb format
├── infer.py                    Script for inference
├── models
│   ├── RRDBNet.py              Generator network definition
│   ├── RaDNet.py               Discriminator network definition
│   └── models.py               Wrapper to instanciate and build models
├── options
│   ├── options.py              Helper script to parse Yaml config files, taken from the mmsr (https://github.com/open-mmlab/mmsr/) repository 
│   └── train_ESRGAN.yml        Config file
├── pretrain.py                 Script for pretraining
├── train.py                    Script for training
├── utils
│   ├── train_util.py           Utility file for training, contains definition of loss functions and optimizers
│   └── util.py                 Utility file, contains methods for image convertion and for making plots
└── validate.py                 Script for validation
```

