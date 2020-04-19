from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Layer, UpSampling2D, LeakyReLU, Input
from tensorflow.keras import Model

# Residual block
class RDB_5C(Layer):
    """
    Represents a Residual Dense Block in the ERSGAN architecture.
    This block consists in 5 forward-connected convolutional layers with leaky ReLU units at the end of the first 4.
    Extends the keras Layer class.
    
    """
    
    def __init__(self, parent_id, id, nf, gc, leak, beta, initializer):
        """
        Initializes the convolutional layers with the specified filter number.
        
        Attributes:
            parent_id: the id of the parent block (RRDB)
            id: the id (position) in the parent block (RRDB)
            nf: number of filters of the (outer) convolutional layers
            gc: growth channel, number of filters of the inner convolutional layers
            leak: leak factor of activation units
            beta: scaling factor for residual blocks
            initializer: kernel initializer for convolutional layers
        """
        
        super(RDB_5C, self).__init__(name='RDB_{0}_{0}'.format(parent_id, id))
        
        self.conv1 = Conv2D(filters=gc, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.conv2 = Conv2D(filters=gc, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.conv3 = Conv2D(filters=gc, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.conv4 = Conv2D(filters=gc, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)
        self.conv5 = Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)  
        
        self.lrelu = LeakyReLU(alpha=leak)   
        self.beta = beta
        
    def call(self, x_in):
        """
        Forward pass through the block.
        
        Args:
            x_in: input of the block
        
        Returns:
            The output of the block
        
        """
        
        x1 = self.conv1(x_in)
        x1 = self.lrelu(x1)
        x2 = self.conv2(tf.concat([x_in,x1],axis=3))
        x2 = self.lrelu(x2)
        x3 = self.conv3(tf.concat([x_in,x1,x2],axis=3))
        x3 = self.lrelu(x3)
        x4 = self.conv4(tf.concat([x_in,x1,x2, x3],axis=3))
        x4 = self.lrelu(x4)
        x5 = self.conv5(tf.concat([x_in,x1,x2, x3, x4],axis=3))
        
        return x_in + x5 * self.beta
    

class RRDB(Layer):
    """
    Represents a Residual-in-Residual Dense Block in the ERSGAN architecture.
    This block consists in 3 successive RDBs with a shortcut path.
    Extends the keras Layer class.
    
    """
    
    def __init__(self, id, nf, gc, leak, beta, initializer):
        """
        Initializes the convolutional layers with the specified filter number.
        
        Attributes:
            id: id (position) of the RRDB in the trunk of the main network
            nf: number of filters of the (outer) convolutional layers
            gc: growth channel, number of filters of the inner convolutional layers
            leak: leak factor of activation units
            beta: scaling factor for residual blocks
            initializer: kernel initializer for convolutional layers
        """
        
        super(RRDB, self).__init__(name='RRDB_{0}'.format(id))
        self.rdb1 = RDB_5C(id, 0, nf, gc, leak, beta, initializer)
        self.rdb2 = RDB_5C(id, 1, nf, gc, leak, beta, initializer)
        self.rdb3 = RDB_5C(id, 2, nf, gc, leak, beta, initializer)
        
        self.beta = beta
        
    def call(self, x_in):
        """
        Forward pass through the block.
        
        Args:
            x_in: input of the block
        
        Returns:
            The output of the block
        
        """
        
        x = self.rdb1(x_in)
        x = self.rdb2(x)
        x = self.rdb3(x)
        
        return x_in + x * self.beta

# ESRGAN generative network
class RRDBNet(Model):
    """
    Represents a generative network in the ERSGAN architecture.
    This network consists in a first convolutional layer followed by a chain of RRDBs, an upsampling phase and a final convolution.
    Extends the keras Model class.
    
    """
    
    def __init__(self, opt):
        """
        Initializes the convolutional layers with the specified filter number.
        
        Attributes:
            opt: the config file
        
        """
        super(RRDBNet, self).__init__()
        
        in_nc = opt['network_G']['in_nc']
        nf = opt['network_G']['nf']
        gc = opt['network_G']['gc']
        out_nc = opt['network_G']['out_nc']
        nb_blocks = opt['network_G']['nb']
        
        leak = opt['network_G']['activation_leak']
        beta = opt['network_G']['residual_scaling']
        
        initializer_scaling = opt['network_G']['initializer_scaling']
        if opt['network_G']['initializer'] == 'MSRA':
            # MSRA variance scaling initializer
            initializer = tf.keras.initializers.VarianceScaling(2.0*initializer_scaling, mode='fan_in', distribution='truncated_normal')
        else:
            # Keras default initializer
            initializer = 'glorot_uniform'
            

        self.conv_first = Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='conv_first')
        self.RRDB_trunk = [RRDB(i, nf, gc, leak, beta, initializer) for i in range(nb_blocks)]
        self.trunk_conv = Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='trunk_conv')
        
        self.upconv1 = UpSampling2D((2,2),data_format='channels_last', interpolation='nearest', name='upconv1')
        
        self.HRconv = Conv2D(filters=nf, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='HRconv')
        self.conv_last = Conv2D(filters=out_nc, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='conv_last')
        
        self.lrelu = LeakyReLU(alpha=leak, name="LeakyReLU_generic")
    
    def call(self, x_in):
        """
        Forward pass through the network.
        
        Args:
            x_in: input of the network
        
        Returns:
            The output of the network
        
        """
        short = self.conv_first(x_in)
        trunk = self.conv_first(x_in)
        for RRDB_n in self.RRDB_trunk:
            trunk = RRDB_n(trunk)
            
        trunk = self.trunk_conv(trunk)
        x = trunk+short
        
        x = self.upconv1(x)
        x = self.lrelu(x)
        
        x = self.HRconv(x)
        x = self.lrelu(x)
        x = self.conv_last(x)
        
        return x