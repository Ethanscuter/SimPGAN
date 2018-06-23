import torch
import torch.nn as nn
import os
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
from model_gan import G12, G21
from model import D1, D2
from torch.nn import init

from evaluate import evaluate


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class Solver():
    def __init__(self, use_reconst_loss, use_labels, num_classes_market, num_classes_grid, beta1, beta2, g_conv_dim,
                 d_conv_dim, train_iters, batch_size, lr, log_step, sample_step, sample_path, model_path,
                 market_loader, grid_loader, sample_probe_loader, sample_gallery_loader):
        self.market_loader = market_loader
        self.grid_loader = grid_loader
        self.sample_probe_loader = sample_probe_loader
        self.sample_gallery_loader = sample_gallery_loader
        self.g12 = None
        self.g21 = None
        self.d1 = None
        self.d2 = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.use_reconst_loss = use_reconst_loss
        self.use_labels = use_labels
        self.num_classes_market = num_classes_market
        self.num_classes_grid = num_classes_grid
        self.beta1 = beta1
        self.beta2 = beta2
        self.g_conv_dim = g_conv_dim
        self.d_conv_dim = d_conv_dim
        self.train_iters = train_iters
        self.batch_size = batch_size
        self.lr = lr
        self.log_step = log_step
        self.sample_step = sample_step
        self.sample_path = sample_path
        self.model_path = model_path
        self.build_model()

    def build_model(self):
        """Builds a generator and a discriminator."""
        self.g12 = G12(conv_dim=self.g_conv_dim)
        init_weights(self.g12, init_type='normal')
        self.g21 = G21(conv_dim=self.g_conv_dim)
        init_weights(self.g21, init_type='normal')
        self.d1 = D1(conv_dim=self.d_conv_dim, use_labels=self.use_labels)
        init_weights(self.d1, init_type='normal')
        self.d2 = D2(conv_dim=self.d_conv_dim, use_labels=self.use_labels)
        init_weights(self.d2, init_type='normal')

        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())

        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            self.d1.cuda()
            self.d2.cuda()

    def merge_images(self, sources, targets, k=10):
        _, _, h, w = sources.shape
        row = int(np.sqrt(self.batch_size))
        merged = np.zeros([3, row * h, row * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            i = idx // row
            j = idx % row
            merged[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            merged[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return merged.transpose(1, 2, 0)

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def to_data(self, x):
        """Converts variable to numpy."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data.numpy()

    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def train(self):
        market_iter = iter(self.market_loader)
        grid_iter = iter(self.grid_loader)
        len_s = len(market_iter)
        len_m = len(grid_iter)
        cur_s = 0
        cur_m = 0

        # loss if use_labels = True
        criterion = nn.CrossEntropyLoss()
        criterion_cycle = nn.L1Loss()

        for step in range(self.train_iters + 1):
            # reset data_iter for each epoch

            if cur_s >= len_s:
                market_iter = iter(self.market_loader)
            if cur_m >= len_m:
                grid_iter = iter(self.grid_loader)

            # load market and grid dataset
            market, m_labels = market_iter.next()
            cur_s = cur_s + 1
            market, m_labels = self.to_var(market), self.to_var(m_labels).long().squeeze()
            grid, g_labels = grid_iter.next()
            cur_m = cur_m + 1
            grid, g_labels = self.to_var(grid), self.to_var(g_labels)

            if self.use_labels:
                grid_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes_market] * market.size(0)).long())
                market_fake_labels = self.to_var(
                    torch.Tensor([self.num_classes_grid] * grid.size(0)).long())

            # ============ train G ============#

            # train grid-market-grid cycle
            self.reset_grad()
            fake_market = self.g12(grid)
            out = self.d2(fake_market)
            reconst_grid = self.g21(fake_market)
            if self.use_labels:
                g_loss_m = criterion(out, m_labels)
            else:
                g_loss_m = torch.mean((out - 1) ** 2)

            if self.use_reconst_loss:
                g_loss_m += torch.mean(torch.abs(grid - reconst_grid))*10

            idt_m = self.g12(market)
            g_loss_m += torch.mean(torch.abs(market - idt_m))*0.5*10

            # train market-grid-market cycle
            self.reset_grad()
            fake_grid = self.g21(market)
            out = self.d1(fake_grid)
            reconst_market = self.g12(fake_grid)
            if self.use_labels:
                g_loss_s = criterion(out, g_labels)
            else:
                g_loss_s = torch.mean((out - 1) ** 2)

            if self.use_reconst_loss:
                g_loss_s += torch.mean(torch.abs(market - reconst_market))*10

            idt_s = self.g21(grid)
            g_loss_s += torch.mean(torch.abs(grid - idt_s)) * 0.5 * 10

            g_loss = g_loss_m + g_loss_s

            g_loss.backward()
            self.g_optimizer.step()

            # ============ train D ============#

            # train with real images
            self.reset_grad()
            out = self.d1(grid)
            if self.use_labels:
                d1_loss = criterion(out, g_labels)
            else:
                d1_loss = torch.mean((out - 1) ** 2)

            out = self.d2(market)
            if self.use_labels:
                d2_loss = criterion(out, m_labels)
            else:
                d2_loss = torch.mean((out - 1) ** 2)

            d_grid_loss = d1_loss
            d_market_loss = d2_loss
            d_real_loss = d1_loss + d2_loss
            if step % 6 == 0:
                d_real_loss.backward()
                self.d_optimizer.step()

            # train with fake images
            self.reset_grad()
            fake_market = self.g12(grid)
            out = self.d2(fake_market)
            if self.use_labels:
                d2_loss = criterion(out, market_fake_labels)
            else:
                d2_loss = torch.mean(out ** 2)

            fake_grid = self.g21(market)
            out = self.d1(fake_grid)
            if self.use_labels:
                d1_loss = criterion(out, grid_fake_labels)
            else:
                d1_loss = torch.mean(out ** 2)

            d_fake_loss = d1_loss + d2_loss

            if step % 6 == 0:
                d_fake_loss.backward()
                self.d_optimizer.step()

            # print the log info
            if (step + 1) % self.log_step == 0:
                print('Step [%d/%d], d_real_loss: %.4f, d_grid_loss: %.4f, d_market_loss: %.4f, '
                      'd_fake_loss: %.4f, g_loss: %.4f'
                      % (step + 1, self.train_iters, d_real_loss.data[0], d_grid_loss.data[0],
                         d_market_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))

            # save the sampled images
            if (step + 1) % self.sample_step == 0:

                fake_market = self.g12(grid)
                fake_grid = self.g21(market)

                grid, fake_grid = self.to_data(grid), self.to_data(fake_grid)
                market, fake_market = self.to_data(market), self.to_data(fake_market)

                merged = self.merge_images(grid, fake_market)
                path = os.path.join(self.sample_path, 'sample-d-g-m.png')
                scipy.misc.imsave(path, merged)
                print('saved %s' % path)

                merged = self.merge_images(market, fake_grid)
                path = os.path.join(self.sample_path, 'sample-d-m-g.png')
                scipy.misc.imsave(path, merged)
                print('saved %s' % path)

            # if (step + 1) % 5000 == 0:
            #     # save the model parameters for each epoch
            #     g12_path = os.path.join(self.model_path, 'g12-%d.pkl' % (step + 1))
            #     g21_path = os.path.join(self.model_path, 'g21-%d.pkl' % (step + 1))
            #     d1_path = os.path.join(self.model_path, 'd1-%d.pkl' % (step + 1))
            #     d2_path = os.path.join(self.model_path, 'd2-%d.pkl' % (step + 1))
            #     torch.save(self.g12.state_dict(), g12_path)
            #     torch.save(self.g21.state_dict(), g21_path)
            #     torch.save(self.d1.state_dict(), d1_path)
            #     torch.save(self.d2.state_dict(), d2_path)

            if (step + 1) % 500 == 0:
                for i, (images, labels) in enumerate(self.sample_probe_loader):
                    grid = self.to_var(images)
                    label = labels[0]
                    path = os.path.join('./samples/probe', label)
                    grid_transfer = self.g12(grid)
                    grid_transfer = self.to_data(grid_transfer)
                    grid_transfer = np.transpose(np.squeeze(grid_transfer), (1, 2, 0))
                    scipy.misc.imsave(path, grid_transfer)

                for i, (images, labels) in enumerate(self.sample_gallery_loader):
                    image = self.to_var(images)
                    label = labels[0]
                    path = os.path.join('./samples/gallery', label)
                    grid_transfer = self.g12(image)
                    grid_transfer = self.to_data(grid_transfer)
                    grid_transfer = np.transpose(np.squeeze(grid_transfer), (1, 2, 0))
                    scipy.misc.imsave(path, grid_transfer)

                evaluate()
