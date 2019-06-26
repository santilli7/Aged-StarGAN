import os
import torch
import torch.nn.functional as F
import numpy as np
import time
from stargan import Generatore
from stargan import Discriminatore
from torch.autograd import Variable
from torchvision.utils import save_image
from logger import Logger


class Model(object):
    """Build StarGAN model."""

    def __init__(self,init):
        self.init = init   

    def build_model(self, init):
        """Crea il generatore e il discriminatore."""
        if init.dataset in ['CelebA', 'RaFD']:
            self.G = Generatore(init.g_conv_dim, init.c_dim, init.g_repeat_num)
            self.D = Discriminatore(init.image_size, init.d_conv_dim, init.c_dim, init.d_repeat_num) 
        elif init.dataset in ['Both']:
            self.G = Generatore(init.g_conv_dim, init.c_dim+init.c2_dim+2, init.g_repeat_num)   # 2 for mask vector.
            self.D = Discriminatore(init.image_size, init.d_conv_dim, init.c_dim+init.c2_dim, init.d_repeat_num)

        self.g_optimizer = torch.optim.Adam(self.G.parameters(), init.g_lr, [init.beta1, init.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), init.d_lr, [init.beta1, init.beta2])
        #self.print_network(self.G, 'G')
        #self.print_network(self.D, 'D')
            
        self.G.to(init.device)
        self.D.to(init.device)
        
        return self

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        self.logger = Logger(self.init.log_dir)

    def restore_model(self, init, fwhere):
        """Restore the trained generator and discriminator."""

        if fwhere == 'train':
            print('Loading the trained models from step {}...'.format(init.resume_iters))
            G_path = os.path.join(init.model_save_dir, '{}-G.ckpt'.format(init.resume_iters))
            D_path = os.path.join(init.model_save_dir, '{}-D.ckpt'.format(init.resume_iters))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        elif fwhere == 'test':
            print('Loading the trained models from step {}...'.format(init.test_iters))
            G_path = os.path.join(init.model_save_dir, '{}-G.ckpt'.format(init.test_iters))
            D_path = os.path.join(init.model_save_dir, '{}-D.ckpt'.format(init.test_iters))
            self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
