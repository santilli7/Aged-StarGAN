from torch.autograd import Variable
from torchvision.utils import save_image
from model import Model as m
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime


class Train(object):
    """Train Stargan model."""

    def __init__(self, init):
        self.init = init
        self.model = m.build_model(m, init)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.model.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.model.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.model.g_optimizer.zero_grad()
        self.model.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.init.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

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
                    # Reverse attribute value.
                    c_trg[:, i] = (c_trg[:, i] == 0)
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.init.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def restore_model(self,init, fwhere):
        """Restore the trained generator and discriminator."""

        if fwhere == 'train':
            print('Loading the trained models from step {}...'.format(init.resume_iters))
            G_path = os.path.join(init.model_save_dir, '{}-G.ckpt'.format(init.resume_iters))
            D_path = os.path.join(init.model_save_dir, '{}-D.ckpt'.format(init.resume_iters))
            self.model.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.model.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
        elif fwhere == 'test':
            print('Loading the trained models from step {}...'.format(init.test_iters))
            G_path = os.path.join(init.model_save_dir, '{}-G.ckpt'.format(init.test_iters))
            D_path = os.path.join(init.model_save_dir, '{}-D.ckpt'.format(init.test_iters))
            self.model.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
            self.model.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        if self.init.dataset == 'CelebA':
            data_loader = self.init.celeba_loader
        elif self.init.dataset == 'RaFD':
            data_loader = self.init.rafd_loader

        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        x_fixed, c_org = next(data_iter)
        x_fixed = x_fixed.to(self.init.device)
        c_fixed_list = self.create_labels(
            c_org, self.init.c_dim, self.init.dataset, self.init.selected_attrs)

        # Learning rate cache for decaying.
        g_lr = self.init.g_lr
        d_lr = self.init.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.init.resume_iters:
            start_iters = self.init.resume_iters
            self.restore_model(self.init,'train')

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.init.num_iters):

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            # Fetch real images and labels.
            try:
                x_real, label_org = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, label_org = next(data_iter)

            # Generate target domain labels randomly.
            rand_idx = torch.randperm(label_org.size(0))
            label_trg = label_org[rand_idx]

            if self.init.dataset == 'CelebA':
                c_org = label_org.clone()
                c_trg = label_trg.clone()
            elif self.init.dataset == 'RaFD':
                c_org = self.label2onehot(label_org, self.init.c_dim)
                c_trg = self.label2onehot(label_trg, self.init.c_dim)

            x_real = x_real.to(self.init.device)           # Input images.
            # Original domain labels.
            c_org = c_org.to(self.init.device)
            # Target domain labels.
            c_trg = c_trg.to(self.init.device)
            # Labels for computing classification loss.
            label_org = label_org.to(self.init.device)
            # Labels for computing classification loss.
            label_trg = label_trg.to(self.init.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.model.D(x_real)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = self.classification_loss(
                out_cls, label_org, self.init.dataset)

            # Compute loss with fake images.
            x_fake = self.model.G(x_real, c_trg)
            out_src, out_cls = self.model.D(x_fake.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.init.device)
            x_hat = (alpha * x_real.data + (1 - alpha)
                     * x_fake.data).requires_grad_(True)
            out_src, _ = self.model.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.init.lambda_cls * \
                d_loss_cls + self.init.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.model.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()
            loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.init.n_critic == 0:
                # Original-to-target domain.
                x_fake = self.model.G(x_real, c_trg)
                out_src, out_cls = self.model.D(x_fake)
                g_loss_fake = - torch.mean(out_src)
                g_loss_cls = self.classification_loss(
                    out_cls, label_trg, self.init.dataset)

                # Target-to-original domain.
                x_reconst = self.model.G(x_fake, c_org)
                g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.init.lambda_rec * \
                    g_loss_rec + self.init.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.model.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.init.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(
                    et, i+1, self.init.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)        

            # Translate fixed images for debugging.
            if (i+1) % self.init.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        x_fake_list.append(self.model.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(
                        self.init.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()),
                               sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.init.model_save_step == 0:
                G_path = os.path.join(
                    self.init.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(
                    self.init.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.model.G.state_dict(), G_path)
                torch.save(self.model.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(
                    self.init.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.init.lr_update_step == 0 and (i+1) > (self.init.num_iters - self.init.num_iters_decay):
                g_lr -= (self.init.g_lr / float(self.init.num_iters_decay))
                d_lr -= (self.init.d_lr / float(self.init.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))

    def train_multi(self):
        """Train StarGAN with multiple datasets."""
        # Data iterators.
        celeba_iter = iter(self.init.celeba_loader)
        rafd_iter = iter(self.init.rafd_loader)

        # Fetch fixed inputs for debugging.
        x_fixed, c_org = next(celeba_iter)
        x_fixed = x_fixed.to(self.init.device)
        c_celeba_list = self.create_labels(
            c_org, self.init.c_dim, 'CelebA', self.init.selected_attrs)
        c_rafd_list = self.create_labels(c_org, self.init.c2_dim, 'RaFD')
        # Zero vector for CelebA.
        zero_celeba = torch.zeros(x_fixed.size(
            0), self.init.c_dim).to(self.init.device)
        # Zero vector for RaFD.
        zero_rafd = torch.zeros(x_fixed.size(
            0), self.init.c2_dim).to(self.init.device)
        # Mask vector: [1, 0].
        mask_celeba = self.label2onehot(torch.zeros(
            x_fixed.size(0)), 2).to(self.init.device)
        # Mask vector: [0, 1].
        mask_rafd = self.label2onehot(torch.ones(
            x_fixed.size(0)), 2).to(self.init.device)

        # Learning rate cache for decaying.
        g_lr = self.init.g_lr
        d_lr = self.init.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.init.resume_iters:
            start_iters = self.init.resume_iters
            self.init.restore_model(self.init,'train')

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.init.num_iters):
            for dataset in ['CelebA', 'RaFD']:

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                data_iter = celeba_iter if dataset == 'CelebA' else rafd_iter

                try:
                    x_real, label_org = next(data_iter)
                except:
                    if dataset == 'CelebA':
                        celeba_iter = iter(self.init.celeba_loader)
                        x_real, label_org = next(celeba_iter)
                    elif dataset == 'RaFD':
                        rafd_iter = iter(self.init.rafd_loader)
                        x_real, label_org = next(rafd_iter)

                # Generate target domain labels randomly.
                rand_idx = torch.randperm(label_org.size(0))
                label_trg = label_org[rand_idx]

                if dataset == 'CelebA':
                    c_org = label_org.clone()
                    c_trg = label_trg.clone()
                    zero = torch.zeros(x_real.size(0), self.init.c2_dim)
                    mask = self.label2onehot(torch.zeros(x_real.size(0)), 2)
                    c_org = torch.cat([c_org, zero, mask], dim=1)
                    c_trg = torch.cat([c_trg, zero, mask], dim=1)
                elif dataset == 'RaFD':
                    c_org = self.label2onehot(label_org, self.init.c2_dim)
                    c_trg = self.label2onehot(label_trg, self.init.c2_dim)
                    zero = torch.zeros(x_real.size(0), self.init.c_dim)
                    mask = self.label2onehot(torch.ones(x_real.size(0)), 2)
                    c_org = torch.cat([zero, c_org, mask], dim=1)
                    c_trg = torch.cat([zero, c_trg, mask], dim=1)

                # Input images.
                x_real = x_real.to(self.init.device)
                # Original domain labels.
                c_org = c_org.to(self.init.device)
                # Target domain labels.
                c_trg = c_trg.to(self.init.device)
                # Labels for computing classification loss.
                label_org = label_org.to(self.init.device)
                # Labels for computing classification loss.
                label_trg = label_trg.to(self.init.device)

                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out_src, out_cls = self.model.D(x_real)
                out_cls = out_cls[:, :self.init.c_dim] if dataset == 'CelebA' else out_cls[:, self.init.c_dim:]
                d_loss_real = - torch.mean(out_src)
                d_loss_cls = self.classification_loss(
                    out_cls, label_org, dataset)

                # Compute loss with fake images.
                x_fake = self.model.G(x_real, c_trg)
                out_src, _ = self.model.D(x_fake.detach())
                d_loss_fake = torch.mean(out_src)

                # Compute loss for gradient penalty.
                alpha = torch.rand(x_real.size(0), 1, 1,
                                   1).to(self.init.device)
                x_hat = (alpha * x_real.data + (1 - alpha)
                         * x_fake.data).requires_grad_(True)
                out_src, _ = self.model.D(x_hat)
                d_loss_gp = self.gradient_penalty(out_src, x_hat)

                # Backward and optimize.
                d_loss = d_loss_real + d_loss_fake + self.init.lambda_cls * \
                    d_loss_cls + self.init.lambda_gp * d_loss_gp
                self.reset_grad()
                d_loss.backward()
                self.model.d_optimizer.step()

                # Logging.
                loss = {}
                loss['D/loss_real'] = d_loss_real.item()
                loss['D/loss_fake'] = d_loss_fake.item()
                loss['D/loss_cls'] = d_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

                # =================================================================================== #
                #                               3. Train the generator                                #
                # =================================================================================== #

                if (i+1) % self.init.n_critic == 0:
                    # Original-to-target domain.
                    x_fake = self.model.G(x_real, c_trg)
                    out_src, out_cls = self.model.D(x_fake)
                    out_cls = out_cls[:,
                                      :self.init.c_dim] if dataset == 'CelebA' else out_cls[:, self.init.c_dim:]
                    g_loss_fake = - torch.mean(out_src)
                    g_loss_cls = self.classification_loss(
                        out_cls, label_trg, dataset)

                    # Target-to-original domain.
                    x_reconst = self.model.G(x_fake, c_org)
                    g_loss_rec = torch.mean(torch.abs(x_real - x_reconst))

                    # Backward and optimize.
                    g_loss = g_loss_fake + self.init.lambda_rec * \
                        g_loss_rec + self.init.lambda_cls * g_loss_cls
                    self.reset_grad()
                    g_loss.backward()
                    self.model.g_optimizer.step()

                    # Logging.
                    loss['G/loss_fake'] = g_loss_fake.item()
                    loss['G/loss_rec'] = g_loss_rec.item()
                    loss['G/loss_cls'] = g_loss_cls.item()

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training info.
                if (i+1) % self.init.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}], Dataset [{}]".format(
                        et, i+1, self.init.num_iters, dataset)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    
            # Translate fixed images for debugging.
            if (i+1) % self.init.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_celeba_list:
                        c_trg = torch.cat(
                            [c_fixed, zero_rafd, mask_celeba], dim=1)
                        x_fake_list.append(self.model.G(x_fixed, c_trg))
                    for c_fixed in c_rafd_list:
                        c_trg = torch.cat(
                            [zero_celeba, c_fixed, mask_rafd], dim=1)
                        x_fake_list.append(self.model.G(x_fixed, c_trg))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(
                        self.init.sample_dir, '{}-images.jpg'.format(i+1))
                    save_image(self.denorm(x_concat.data.cpu()),
                               sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.init.model_save_step == 0:
                G_path = os.path.join(
                    self.init.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(
                    self.init.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.model.G.state_dict(), G_path)
                torch.save(self.model.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(
                    self.init.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.init.lr_update_step == 0 and (i+1) > (self.init.num_iters - self.init.num_iters_decay):
                g_lr -= (self.init.g_lr / float(self.init.num_iters_decay))
                d_lr -= (self.init.d_lr / float(self.init.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))
