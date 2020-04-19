from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import math
import glob
import argparse


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from options.options import parse
from utils import train_util, util
from models.models import make_generator, make_discriminator

from data_scripts import data_loader
from data_scripts.data_loader import DatasetLoader, decode_img_from_path

def train(opt):
    """
    Train the model using a GAN framework and all losses and save the model after training.
    
    Args:
        opt: the options file

    """
    
    # Configure GPUS
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

    # Training constants
    n_epochs = opt['train']['n_epochs']
    n_patches = opt['datasets']['train']['n_images']*(opt['datasets']['train']['LR_size']/opt['datasets']['train']['LR_stride']-1)**2
    batches_per_epoch = int(math.ceil(n_patches / opt['datasets']['train']['batch_size']))
    batch_size_per_device = int(opt['datasets']['train']['batch_size']/max(1,len(gpus)))
    decay_steps = [batches_per_epoch*i for i in range(n_epochs)]
    
    # Enable distributed computation
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # build generator and discriminator
        model = make_generator(opt, print_summary=True)
        discriminator = make_discriminator(opt, print_summary=True)
        
        checkpoints = sorted(glob.glob(opt['path']['checkpoint_root']+"*.h5"))
        start_epoch=0
        if opt['path']['resume'] and len(checkpoints) > 0:
            last_checkpoint = checkpoints[-1]
            path = last_checkpoint
            start_epoch = int(re.findall(r'\d+', last_checkpoint)[-2])# -1 is h5
            path_disc = sorted(glob.glob(opt['path']['checkpoint_root_disc']+"*.h5"))[-1]
            discriminator.load_weights(path_disc)
            print("Resuming training at epoch "+str(start_epoch)+" from checkpoint: "+path)
        elif opt['path']['resume'] and os.path.exists(opt['path']['pretrained_model_G']):
            model.load_weights(opt['path']['pretrained_model_G'])
            print("Started training with pretrained model: "+opt['path']['pretrained_model_G'])
        else:
            print("No pretrained model found, training from the beginning")
        
    
        # train generator with ESRGAN generator loss function
        loss_func_G = train_util.GeneratorLoss(batch_size_per_device, opt)
        pix_loss_fn = train_util.PixelLoss(batch_size_per_device, opt)
        feat_loss_fn = train_util.FeatureLoss(batch_size_per_device, opt)
        
        optimizer_G = train_util.generator_optimizer(opt,decay_steps)
        
        # train discriminator with discriminator loss function
        optimizer_D = train_util.discriminator_optimizer(opt,decay_steps)
    
        @tf.function
        def train_step(lr, hr):
            def step_fn(lr,hr):
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # Feed the batch to the network
                    out = model(lr, training=True)
                    
                    # Compute losses on the output batch
                    pix_loss_G = pix_loss_fn(out,hr)
                    feat_loss_G = feat_loss_fn(out,hr)
                        
                    out_disc = discriminator(out, training=True)
                    hr_disc = discriminator(hr, training=True)
                    
                    adv_loss_G, adv_loss_D = train_util.get_adversarial_loss(out_disc, hr_disc)


                    loss_G = opt['train']['pixel_weight'] * pix_loss_G + opt['train']['feature_weight'] * feat_loss_G + opt['train']['gan_weight']*adv_loss_G
                    loss_D = adv_loss_D

                # Apply optimization step
                gradients_G = gen_tape.gradient(loss_G, model.trainable_variables)
                optimizer_G.apply_gradients(zip(gradients_G, model.trainable_variables))
                
                gradients_D = disc_tape.gradient(loss_D, discriminator.trainable_variables)
                optimizer_D.apply_gradients(zip(gradients_D, discriminator.trainable_variables))
                
                return loss_G

            # Compute batch loss (for logging)
            per_example_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(lr,hr,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=None)
            return mean_loss


        loader = DatasetLoader(opt)
        print("Dataset built")

        checkpoint_name = opt['path']['checkpoint_root']+opt['path']['checkpoint_name_train']
        checkpoint_name_disc = opt['path']['checkpoint_root_disc']+opt['path']['checkpoint_name_disc']
        for e in range(start_epoch,  n_epochs):
            print("e:%d/%d"%(e+1, n_epochs))
            # load data
            dataset = loader.next(opt)
            dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
            # For each batch in the distributed dataset
            for lr_batch, hr_batch in dist_dataset:
                # Feed the batch to the model
                loss = train_step(lr_batch, hr_batch)
            tf.print("loss:",loss)
    
            # Save new checkpoint
            model.save_weights(checkpoint_name+"_e"+str(e+1)+".h5")
            util.remove_if_present(checkpoint_name+"_e"+str(e)+".h5")
            discriminator.save_weights(checkpoint_name_disc+"_e"+str(e+1)+".h5")
            util.remove_if_present(checkpoint_name_disc+"_e"+str(e)+".h5")
    
        #Save final model
        model.save_weights(opt['path']['trained_model_G'])
        print("Saved model to ",opt['path']['trained_model_G'])
        util.remove_if_present(checkpoint_name+"_e"+str(e+1)+".h5")
        util.remove_if_present(checkpoint_name_disc+"_e"+str(e+1)+".h5")
        
def main():
    ## options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    if args.opt is not None:
        opt = parse(args.opt)
    else:
        opt = parse('./options/train_ESRGAN.yml')

    train(opt)
    
    
if __name__ == '__main__':
    main()
