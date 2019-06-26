from torch.autograd import Variable
from torchvision.utils import save_image
from model import Model as m
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime

class Test(object):
    """Testing StarGAN"""

    def __init__(self, init):
        self.init = init
        self.model = m
        self.resume_iters = init.test_iters

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.init.device))
        return c_trg_list

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out
    
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.model.restore_model(self.model, self.init,'test')
        
        # Set data loader.
        if self.init.dataset == 'CelebA':
            data_loader = self.init.celeba_loader
        elif self.init.dataset == 'RaFD':
            data_loader = self.init.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.init.device)
                c_trg_list = self.create_labels(c_org, self.init.c_dim, self.init.dataset, self.init.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                   # print('XCONCAT ...',self.model.G(x_real, c_trg))
                    x_fake_list.append(self.model.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                #print('XCONCAT ...',x_concat)
                #print('XCONCAT ...',x_fake_list.count)
                result_path = os.path.join(self.init.result_dir, '{}-images.jpg'.format(i+1))
                #print('IMAGE ...', self.model.denorm(x_concat[1].data.cpu()))
                #save_image(self.model.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                save_image(self.denorm(x_concat[1].data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))

    def test_multi(self):
        """Translate images using StarGAN trained on multiple datasets."""
        # Load the trained generator.
        self.model.restore_model(self.model,self.init.test_iters,'test')
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(self.init.celeba_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.init.device)
                c_celeba_list = self.create_labels(c_org, self.init.c_dim, 'CelebA', self.init.selected_attrs)
                c_rafd_list = self.create_labels(c_org, self.init.c2_dim, 'RaFD')
                zero_celeba = torch.zeros(x_real.size(0), self.init.c_dim).to(self.init.device)      # Zero vector for CelebA.
                zero_rafd = torch.zeros(x_real.size(0), self.init.c2_dim).to(self.init.device)       # Zero vector for RaFD.
                mask_celeba = self.label2onehot(torch.zeros(x_real.size(0)), 2).to(self.init.device)  # Mask vector: [1, 0].
                mask_rafd = self.label2onehot(torch.ones(x_real.size(0)), 2).to(self.init.device)     # Mask vector: [0, 1].

                # Translate images.
                x_fake_list = [x_real]
                for c_celeba in c_celeba_list:
                    c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
                    x_fake_list.append(self.model.G(x_real, c_trg))
                for c_rafd in c_rafd_list:
                    c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
                    x_fake_list.append(self.model.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.init.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))