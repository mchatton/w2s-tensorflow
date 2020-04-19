from __future__ import absolute_import, division, print_function, unicode_literals

import os
import re
import math
import glob
import argparse


from tqdm import tqdm
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from options.options import parse
from utils import train_util, util
from models.models import make_generator
from data_scripts.data_loader import DatasetLoader

def pretrain(opt):
    """
    Train the model using only pixel-wise loss and save the model after pretraining.
    
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

    #Training constants
    n_epochs = opt['train']['n_epochs']
    n_patches = opt['datasets']['train']['n_images']*(opt['datasets']['train']['LR_size']/opt['datasets']['train']['LR_stride']-1)**2
    batches_per_epoch = int(math.ceil(n_patches / opt['datasets']['train']['batch_size']))
    batch_size_per_device = int(opt['datasets']['train']['batch_size']/max(1,len(gpus)))
    print(batches_per_epoch)
    decay_steps = [batches_per_epoch*i for i in range(n_epochs)]
    print(decay_steps)
    
    
    # Enable distributed computation
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        # build generator
        model = make_generator(opt, print_summary=True)
        checkpoints = sorted(glob.glob(opt['path']['checkpoint_root']+"*.h5"))
        start_epoch=0
        if opt['path']['resume'] and len(checkpoints) > 0:
            last_checkpoint = checkpoints[-1] 
            path = last_checkpoint
            start_epoch = int(re.findall(r'\d+', last_checkpoint)[-2])# -1 is h5
            model.load_weights(path)
            print("Resuming training at epoch "+str(start_epoch)+" from checkpoint: "+path)
    
        # configure loss and optimizer
        loss_func = train_util.PixelLoss(batch_size_per_device, opt)
        optimizer = train_util.generator_optimizer(opt,decay_steps)
    
        @tf.function
        def pretrain_step(lr, hr):
            def step_fn(lr,hr):
                with tf.GradientTape() as tape:
                    
                    # Feed the batch to the network
                    out = model(lr, training=True)
                    
                    # Compute loss on the output batch
                    loss = loss_func(out,hr)
                    
                # Apply optimization step
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients,model.trainable_variables))
                return loss

            # Compute batch loss (for logging)
            per_example_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(lr,hr,))
            mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=None)
            return mean_loss


        loader = DatasetLoader(opt)


        checkpoint_name = opt['path']['checkpoint_root']+opt['path']['checkpoint_name_pretrain']
        for e in range(start_epoch,  n_epochs):
            print("e:%d/%d"%(e+1, n_epochs))
            # load data
            dataset = loader.next(opt)
            dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)
            # For each batch in the dataset
            for lr_batch, hr_batch in dist_dataset:
                # Feed the batch to the model
                loss = pretrain_step(lr_batch, hr_batch)
                step += 1
                
            tf.print("loss:",loss)
            
            # Save new checkpoint
            model.save_weights(checkpoint_name+"_e"+str(e+1)+".h5")
            util.remove_if_present(checkpoint_name+"_e"+str(e)+".h5")
        
        # Save final model
        model.save_weights(opt['path']['pretrained_model_G'])
        print("Saved model to ",opt['path']['pretrained_model_G'])
        util.remove_if_present(checkpoint_name+"_e"+str(e+1)+".h5")
        
def main():
    ## options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    if args.opt is not None:
        opt = parse(args.opt)
    else:
        opt = parse('./options/train_ESRGAN.yml')
    
    if not os.path.exists(opt['path']['pretrained_model_G']):
        print("Pretrain checkpoint not found, starting pretraining...")
        pretrain(opt)
    else: 
        print("Pretrained model already found: "+opt['path']['pretrained_model_G']+", aborting")
    
    
if __name__ == '__main__':
    main()