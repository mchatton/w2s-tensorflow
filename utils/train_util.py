import tensorflow as tf
    
    
##################
# Loss functions #
##################

class PixelLoss:
    """
    Representation of a pixelwise loss function
    
    """
    
    def __init__(self, batch_size, opt):
        """
        Initializes the loss function.
        
        Attributes:
            batch_size: the batch size of a network instance (batch size on a device) (unused)
            opt: the config file
        
        """
        self.batch_size = batch_size
        self.loss_fn = None
        if opt['train']['pixel_criterion'] == 'l1':
            self.loss_fn = tf.losses.MeanAbsoluteError(reduction=tf.losses.Reduction.SUM)
        elif opt['train']['pixel_criterion'] == 'l2':
            self.loss_fn = tf.losses.MeanSquaredError(reduction=tf.losses.Reduction.SUM)
        else:
            raise ValueError("Only l1 or l2 supported for pixel loss")
            
    def __call__(self,out,hr):
        """
        Compute the loss
        
        Args:
            out: the computed image batch
            hr: the groundtruth image batch
        Returns:
            The pixel loss
        
        """
        return self.loss_fn(out, hr) / out.shape[0]

class FeatureLoss:
    """
    Representation of a content (feature) loss function that is 
    the loss extracted from a classifier model
    
    """
    
    def __init__(self, batch_size, opt):
        """
        Initializes the loss function.
        
        Attributes:
            batch_size: the batch size of a network instance (batch size on a device)
            opt: the config file
        
        """
        self.batch_size = batch_size
        self.use_gram = opt['train']['use_gram']
        self.loss_fn = None
        if opt['train']['feature_criterion'] == 'l1':
            self.loss_fn = tf.losses.MeanAbsoluteError(reduction=tf.losses.Reduction.SUM)
        elif opt['train']['feature_criterion'] == 'l2':
            self.loss_fn = tf.losses.MeanSquaredError(reduction=tf.losses.Reduction.SUM)
        else:
            raise ValueError("Only l1 or l2 supported for feature loss")
        
        hr_size = opt['datasets']['train']['HR_patch_size']
        self.layer_name = opt['train']['feature_layer']
        if opt['train']['feature_model'] == 'VGG19':
            self.model = tf.keras.applications.VGG19(include_top=False, input_shape=(hr_size,hr_size,3)) # RGB mandatory
        else:
            raise ValueError("Only VGG19 supported for feature loss extraction")
            
        if self.model.get_layer(self.layer_name) is None:
            raise ValueError("layer not found in network")
            
        self.model.build((batch_size, hr_size, hr_size,3)) # RGB mandatory
        self.intermediate_layer_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer(self.layer_name).output)

    def __call__(self,out,hr):
        """
        Compute the loss
        
        Args:
            out: the computed image batch
            hr: the groundtruth image batch
        Returns:
            The feature loss
        """
        out = tf.image.grayscale_to_rgb(out)
        hr = tf.image.grayscale_to_rgb(hr)
        
        out_features = self.intermediate_layer_model(out)
        hr_features = self.intermediate_layer_model(hr)
        
        normalize_coeff = out.shape[0]
        if(self.use_gram):
            n = out_features.shape[1]*out_features.shape[2]
            m = out_features.shape[3]
            
            out_features = tf.reshape(out_features, [-1, n, m])  #bxNxM
            out_features = tf.matmul(out_features, out_features, transpose_b=True) #bxNxN
            
            hr_features = tf.reshape(hr_features, [-1, n, m]) #bxNxM
            hr_features = tf.matmul(hr_features, hr_features, transpose_b=True) #bxNxN
            
        return self.loss_fn(out_features, hr_features) / normalize_coeff

class GeneratorLoss:
    """
    Representation of the loss function of the generator model. 
    It consists in a weighted sum of a pixelwise loss and a feature loss.
    
    """
    def __init__(self, batch_size, opt):
        """
        Initializes the loss function.
        
        Attributes:
            batch_size: the batch size of a network instance (batch size on a device)
            opt: the config file
        
        """
        self.pixel_weight = opt['train']['pixel_weight']
        self.pixel_loss = PixelLoss(batch_size, opt)
        self.feature_weight = opt['train']['feature_weight']
        self.feature_loss = FeatureLoss(batch_size, opt)

    def __call__(self,out,hr):

        return  self.pixel_weight * self.pixel_loss(out,hr) + \
                self.feature_weight * self.feature_loss(out,hr)

def get_adversarial_loss(out, hr):
    """
    Compute the adversarial loss of a batch of outputs. Follows the relativistic gan formula of the ESRGAN paper
    
    Args:
        out: the computed image batch
        hr: the groundtruth image batch
    Returns:
        loss_G: the loss for the generator
        loss_D: the loss for the discriminator
    """
    fake_mean = tf.math.reduce_mean(out)
    real_mean = tf.math.reduce_mean(hr)
    
    loss_D = tf.keras.losses.binary_crossentropy(tf.ones_like(out),hr-fake_mean) + tf.keras.losses.binary_crossentropy(tf.zeros_like(hr),out-real_mean)
    loss_G = tf.keras.losses.binary_crossentropy(tf.zeros_like(hr),hr-fake_mean) + tf.keras.losses.binary_crossentropy(tf.ones_like(out),out-real_mean)

    #This is strictly equivalent to the ESRGAN formula, that can be found in the following lines, but is not numerically stable
    #fake_activations = tf.math.sigmoid(out-real_mean)
    #real_activations = tf.math.sigmoid(hr-fake_mean)
    #loss_G = -tf.math.reduce_mean(tf.math.log(fake_activations)) - tf.math.reduce_mean(tf.math.log(1-real_activations))
    #loss_D = -tf.math.reduce_mean(tf.math.log(real_activations)) - tf.math.reduce_mean(tf.math.log(1-fake_activations))
    
    return loss_G, loss_D

##############
# Optimizers #
##############

def generator_optimizer(opt, decay_steps=None):
    """
    Creates the generator optimizer function
    
    Args:
        opt: the config file
        decay_steps (optional): the (optimizer) iterations at which the learning rate decays. If not specified, the learning rate is constant 
    Returns:
        The optimizer function (instance of tf.keras.optimizer)
    """
    if opt['train']['optimizer'] == 'Adam':
        lr_init = opt['train']['lr_G']
        beta1 = opt['train']['beta1_G']
        beta2 = opt['train']['beta2_G']
        ams = opt['train']['use_amsgrad']
        
        if decay_steps:
            lr_gamma = opt['train']['lr_gamma']
            lr_values = [lr_init*(lr_gamma**i) for i in range(len(decay_steps)+1)]
            lr_func = tf.keras.optimizers.schedules.PiecewiseConstantDecay(decay_steps, lr_values)
        
            return tf.keras.optimizers.Adam(learning_rate=lr_func, beta_1=beta1, beta_2=beta2, amsgrad=ams)
        else:
        
            return tf.keras.optimizers.Adam(learning_rate=lr_init, beta_1=beta1, beta_2=beta2, amsgrad=ams)
    else:
        raise ValueError("Only Adam optimizer supported")
        
def discriminator_optimizer(opt, decay_steps=None):
    """
    Creates the discriminator optimizer function
    
    Args:
        opt: the config file
        decay_steps (optional): the (optimizer) iterations at which the learning rate decays. If not specified, the learning rate is constant 
    Returns:
        The optimizer function (instance of tf.keras.optimizer)
    """
    if opt['train']['optimizer'] == 'Adam':
        lr_init = opt['train']['lr_D']
        beta1 = opt['train']['beta1_D']
        beta2 = opt['train']['beta2_D']
        ams = opt['train']['use_amsgrad']
        
        if decay_steps:
            lr_gamma = opt['train']['lr_gamma']
            print(decay_steps)
            
            lr_values = [lr_init*(lr_gamma**i) for i in range(len(decay_steps)+1)]
            print(lr_values)
            lr_func = tf.keras.optimizers.schedules.PiecewiseConstantDecay(decay_steps, lr_values)
        
            return tf.keras.optimizers.Adam(learning_rate=lr_func, beta_1=beta1, beta_2=beta2, amsgrad=ams)
        else:
        
            return tf.keras.optimizers.Adam(learning_rate=lr_init, beta_1=beta1, beta_2=beta2, amsgrad=ams)
    else:
        raise ValueError("Only Adam optimizer supported")