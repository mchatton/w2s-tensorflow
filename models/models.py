from models.RRDBNet import RRDBNet
from models.RaDNet import RaDNet


def make_generator(opt, build=True, print_summary=False):
    """
    Creates and build the generator model
    
    Args:
        opt: the config file
        build: whether or not to build the model with provided options
        print_summary: whether to print the model summary after build
    """
    
    if(opt['network_G']['which_model_G'] == 'RRDBNet'):
        model = RRDBNet(opt)
        if build:
            nc = int(opt['network_G']['in_nc'])
            batch_size = int(opt['datasets']['train']['batch_size'])
            lr_size = int(opt['datasets']['train']['LR_patch_size'])
            model.build((batch_size, lr_size, lr_size, nc))
            if print_summary:
                print(model.summary())
        return model
    else:
        raise ValueError("only RRDBNet supported")
    
def make_discriminator(opt, build=True, print_summary=False):
    """
    Creates and build the discriminator model
    
    Args:
        opt: the config file
        build: whether or not to build the model with provided options
        print_summary: whether to print the model summary after build
    """
    
    if(opt['network_D']['which_model_D'] == 'RaDNet'):
        model = RaDNet(opt)
        if build:
            nc = int(opt['network_D']['in_nc'])
            batch_size = int(opt['datasets']['train']['batch_size'])
            hr_size = int(opt['datasets']['train']['HR_patch_size'])
            model.build((batch_size, hr_size, hr_size, nc))
            if print_summary:
                print(model.summary())
        return model
    else:
        raise ValueError("only RaDNet supported")