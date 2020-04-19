from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Layer, UpSampling2D, LeakyReLU, BatchNormalization, Dense, Flatten
from tensorflow.keras import Model


class RaDNet(Model):
    """
    Represents a discriminator network in the ERSGAN architecture.
    This network follows the one described in the paper, implemented as a VGG128 network
    Extends the keras Model class.
    
    """
    def __init__(self,opt):
        """
        Initializes the discriminator architecture.
        
        Attributes:
            opt: the config file
        
        """
        super(RaDNet, self).__init__()
        
        in_nc = opt['network_D']['in_nc']
        nf = opt['network_D']['nf']
        
        leak = opt['network_G']['activation_leak']
        beta = opt['network_G']['residual_scaling']
        
        self.conv0_0 = Conv2D(nf, kernel_size=3, strides=1, padding='same', use_bias=True, name='conv0_0')
        self.conv0_1 = Conv2D(nf, kernel_size=4, strides=2, padding='same', use_bias=False, name='conv0_1')
        self.bn0_1 = BatchNormalization(name='bn0_1')
        
        self.conv1_0 = Conv2D(nf*2, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv1_0')
        self.bn1_0 = BatchNormalization(name='bn1_0')
        self.conv1_1 = Conv2D(nf*2, kernel_size=4, strides=2, padding='same', use_bias=False, name='conv1_1')
        self.bn1_1 = BatchNormalization(name='bn1_1')
        
        self.conv2_0 = Conv2D(nf*4, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv2_0')
        self.bn2_0 = BatchNormalization(name='bn2_0')
        self.conv2_1 = Conv2D(nf*4, kernel_size=4, strides=2, padding='same', use_bias=False, name='conv2_1')
        self.bn2_1 = BatchNormalization(name='bn2_1')
        
        self.conv3_0 = Conv2D(nf*8, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv3_0')
        self.bn3_0 = BatchNormalization(name='bn3_0')
        self.conv3_1 = Conv2D(nf*8, kernel_size=4, strides=2, padding='same', use_bias=False, name='conv3_1')
        self.bn3_1 = BatchNormalization(name='bn3_1')
        
        self.conv4_0 = Conv2D(nf*8, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv4_0')
        self.bn4_0 = BatchNormalization(name='bn4_0')
        self.conv4_1 = Conv2D(nf*8, kernel_size=4, strides=2, padding='same', use_bias=False, name='conv4_1')
        self.bn4_1 = BatchNormalization(name='bn4_1')
        
        self.linear1 = Dense(100)
        self.linear2 = Dense(1)
        
        self.lrelu = LeakyReLU(alpha=leak)
        self.flatten = Flatten()
        
    def call(self,x):
        """
        Forward pass through the network.
        
        Args:
            x: input of the network
        
        Returns:
            The output of the network
        
        """
        out = self.lrelu(self.conv0_0(x))
        out = self.lrelu(self.bn0_1(self.conv0_1(out)))
    
        out = self.lrelu(self.bn1_0(self.conv1_0(out)))
        out = self.lrelu(self.bn1_1(self.conv1_1(out)))
    
        out = self.lrelu(self.bn2_0(self.conv2_0(out)))
        out = self.lrelu(self.bn2_1(self.conv2_1(out)))
    
        out = self.lrelu(self.bn3_0(self.conv3_0(out)))
        out = self.lrelu(self.bn3_1(self.conv3_1(out)))
    
        out = self.lrelu(self.bn4_0(self.conv4_0(out)))
        out = self.lrelu(self.bn4_1(self.conv4_1(out)))
    
        out = self.flatten(out)
        out = self.lrelu(self.linear1(out))
        out = self.linear2(out)
        return tf.squeeze(out)