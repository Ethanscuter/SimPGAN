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
from model_reid import DSiamese
from eval import evaluate, evaluate_idt

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
                 d_conv_dim, train_iters,
                 batch_size, lr, log_step, sample_step, sample_path, model_path, market_loader, grid_loader,
                 sample_probe_loader, sample_gallery_loader):
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
        self.dr_optimizer = None
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
        self.num_classes_market = num_classes_market
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
        self.dreid = DSiamese(class_count=self.num_classes_market)

        g_params = list(self.g12.parameters()) + list(self.g21.parameters())
        d_params = list(self.d1.parameters()) + list(self.d2.parameters())
        dr_params = list(self.dreid.parameters())

        self.g_optimizer = optim.Adam(g_params, self.lr, [self.beta1, self.beta2])
        self.d_optimizer = optim.Adam(d_params, self.lr, [self.beta1, self.beta2])
        self.dr_optimizer = optim.Adam(dr_params, self.lr, [self.beta1, self.beta2])

        if torch.cuda.is_available():
            self.g12.cuda()
            self.g21.cuda()
            self.d1.cuda()
            self.d2.cuda()
            self.dreid.cuda()

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
        self.dr_optimizer.zero_grad()

    def train(self):
        market_iter = iter(self.market_loader)
        grid_iter = iter(self.grid_loader)
        len_s = len(market_iter)
        len_m = len(grid_iter)
        cur_s = 0
        cur_m = 0

        criterion = nn.CrossEntropyLoss()

        for step in range(self.train_iters + 1):
            # reset data_iter for each epoch

            if cur_s >= len_s:
                market_iter = iter(self.market_loader)
            if cur_m >= len_m:
                grid_iter = iter(self.grid_loader)

            # market dataset: img_left, img_right, target_left, target_right, target_binary
            market_left, market_right, market_label_left, market_label_right, market_label_binary = market_iter.next()
            cur_s = cur_s + 1
            market_left, market_right, market_label_left, market_label_right, market_label_binary = self.to_var(market_left), \
                self.to_var(market_right), self.to_var(market_label_left).long().squeeze(), \
                self.to_var(market_label_right).long().squeeze(), self.to_var(market_label_binary).long().squeeze()
            # grid dataset: img_left, img_right, target_left, target_right, target_binary
            grid_left, grid_right, grid_label_left, grid_label_right, grid_label_binary = grid_iter.next()
            cur_m = cur_m + 1
            grid_left, grid_right, grid_label_left, grid_label_right, grid_label_binary = self.to_var(grid_left), \
                self.to_var(grid_right), self.to_var(grid_label_left).long().squeeze(), \
                self.to_var(grid_label_right).long().squeeze(), self.to_var(grid_label_binary).long().squeeze()

            # ============ train G ============#

            """
            train grid-market-grid cycle: market_left, market_right, grid_left, grid_right
            """
            self.reset_grad()
            fake_market_left = self.g12(grid_left)
            out_left = self.d2(fake_market_left)
            reconst_grid_left = self.g21(fake_market_left)

            fake_market_right = self.g12(grid_right)
            out_right = self.d2(fake_market_right)
            reconst_grid_right = self.g21(fake_market_right)

            # Generator loss A -> B
            g_loss_m = torch.mean((out_left - 1) ** 2)
            g_loss_m += torch.mean((out_right - 1) ** 2)

            # Cycle A -> B -> A
            if self.use_reconst_loss:
                g_loss_m += torch.mean(torch.abs(grid_left - reconst_grid_left)) * 10
                g_loss_m += torch.mean(torch.abs(grid_right - reconst_grid_right)) * 10

            # Identity loss fed real B
            idt_m_left = self.g12(market_left)
            idt_m_right = self.g12(market_right)
            g_loss_m += torch.mean(torch.abs(market_left - idt_m_left)) * 0.5 * 10
            g_loss_m += torch.mean(torch.abs(market_right - idt_m_right)) * 0.5 * 10

            """
            train market-grid-market cycle: market_left, market_right, grid_left, grid_right
            """
            # self.reset_grad()
            fake_grid_left = self.g21(market_left)
            out_left = self.d1(fake_grid_left)
            reconst_market_left = self.g12(fake_grid_left)

            fake_grid_right = self.g21(market_right)
            out_right = self.d1(fake_grid_right)
            reconst_market_right = self.g12(fake_grid_right)

            # Generator loss B -> A
            g_loss_s = torch.mean((out_left - 1) ** 2)
            g_loss_s += torch.mean((out_right - 1) ** 2)

            # Cycle B -> A -> B
            if self.use_reconst_loss:
                g_loss_s += torch.mean(torch.abs(market_left - reconst_market_left)) * 10
                g_loss_s += torch.mean(torch.abs(market_right - reconst_market_right)) * 10

            # Identity loss fed real A
            idt_s_left = self.g21(grid_left)
            idt_s_right = self.g21(grid_right)

            g_loss_s += torch.mean(torch.abs(grid_left - idt_s_left)) * 0.5 * 10
            g_loss_s += torch.mean(torch.abs(grid_right - idt_s_right)) * 0.5 * 10

            g_loss = g_loss_m + g_loss_s

            g_loss.backward()
            self.g_optimizer.step()

            # ============ train D ============#
            '''
            train with real images: market_left, market_right, grid_left, grid_right
            '''
            # Train D1 with real image
            self.reset_grad()
            out_left = self.d1(grid_left)
            out_right = self.d1(grid_right)

            d1_loss = torch.mean((out_left - 1) ** 2)
            d1_loss += torch.mean((out_right - 1) ** 2)

            # Train D2 with real image
            out_left = self.d2(market_left)
            out_right = self.d2(market_right)

            d2_loss = torch.mean((out_left - 1) ** 2)
            d2_loss += torch.mean((out_right - 1) ** 2)

            # in order to print
            d_grid_loss = d1_loss
            d_market_loss = d2_loss
            d_real_loss = d1_loss + d2_loss
            if step % 6 == 0:
                d_real_loss.backward()
                self.d_optimizer.step()

            '''
            train with fake images: 
            '''
            # Train D2 loss with fake image
            self.reset_grad()
            fake_market_left = self.g12(grid_left)
            fake_market_right = self.g12(grid_right)

            out_left = self.d2(fake_market_left)
            out_right = self.d2(fake_market_right)

            d2_loss = torch.mean(out_left ** 2)
            d2_loss += torch.mean(out_right ** 2)

            # Train D1 loss with fake image
            fake_grid_left = self.g21(market_left)
            fake_grid_right = self.g21(market_right)
            out_left = self.d1(fake_grid_left)
            out_right = self.d1(fake_grid_right)
            d1_loss = torch.mean(out_left ** 2)
            d1_loss += torch.mean(out_right ** 2)

            d_fake_loss = d1_loss + d2_loss

            if step % 6 == 0:
                d_fake_loss.backward()
                self.d_optimizer.step()

            '''
            Reid consistant: 
            '''
            if step >= 500:
                # Reid Identify loss G21(A) -> fake A
                self.reset_grad()
                fake_market_left = self.g12(market_left)
                fake_market_right = self.g12(market_right)
                out_idt1, out_idt2, out_bi = self.dreid(fake_market_left, fake_market_right)
                dreid_loss_idt = criterion(out_idt1, market_label_left)
                dreid_loss_idt += criterion(out_idt2, market_label_right)
                dreid_loss_idt += criterion(out_bi, market_label_binary)

            # Reid Cycle loss G21(G12(A)) -> fake A
                fake_grid_left = self.g21(market_left)
                fake_market_left = self.g12(fake_grid_left)

                fake_grid_right = self.g21(market_right)
                fake_market_right = self.g12(fake_grid_right)

                out_idt1, out_idt2, out_bi = self.dreid(fake_market_left, fake_market_right)
                dreid_loss_cycle = criterion(out_idt1, market_label_left)
                dreid_loss_cycle += criterion(out_idt2, market_label_right)
                dreid_loss_cycle += criterion(out_bi, market_label_binary)

            # Reid loss with real image
                out1, out2, out_bi = self.dreid(market_left, market_right)
                dreid_loss_real = criterion(out1, market_label_left)
                dreid_loss_real += criterion(out2, market_label_right)
                dreid_loss_real += criterion(out_bi, market_label_binary)

                dreid_loss = dreid_loss_idt + dreid_loss_cycle + dreid_loss_real

                if step % 6 == 0:
                    dreid_loss.backward()
                    self.d_optimizer.step()

            if step >= 3000:
                # Reid Identify loss G21(A) -> fake A
                self.reset_grad()
                fake_market_left = self.g12(market_left)
                fake_market_right = self.g12(market_right)
                out_idt1, out_idt2, out_bi = self.dreid(fake_market_left, fake_market_right)
                dreid_loss_idt = criterion(out_idt1, market_label_left)
                dreid_loss_idt += criterion(out_idt2, market_label_right)
                dreid_loss_idt += criterion(out_bi, market_label_binary)

                # Reid Cycle loss G21(G12(A)) -> fake A
                fake_grid_left = self.g21(market_left)
                fake_market_left = self.g12(fake_grid_left)

                fake_grid_right = self.g21(market_right)
                fake_market_right = self.g12(fake_grid_right)

                out_idt1, out_idt2, out_bi = self.dreid(fake_market_left, fake_market_right)
                dreid_loss_cycle = criterion(out_idt1, market_label_left)
                dreid_loss_cycle += criterion(out_idt2, market_label_right)
                dreid_loss_cycle += criterion(out_bi, market_label_binary)

                # Reid loss with real image
                out1, out2, out_bi = self.dreid(market_left, market_right)
                dreid_loss_real = criterion(out1, market_label_left)
                dreid_loss_real += criterion(out2, market_label_right)
                dreid_loss_real += criterion(out_bi, market_label_binary)

                dreid_loss = dreid_loss_idt + dreid_loss_cycle + dreid_loss_real

                dreid_loss = dreid_loss / 6

                # if step % 6 == 0:
                dreid_loss.backward()
                self.dr_optimizer.step()

            # print the log info
            if step < 500:
                if (step + 1) % self.log_step == 0:
                    print('Step [%d/%d], d_real_loss: %.4f, d_grid_loss: %.4f, d_market_loss: %.4f, '
                          'd_fake_loss: %.4f, g_loss: %.4f, '
                          % (step + 1, self.train_iters, d_real_loss.data[0], d_grid_loss.data[0],
                             d_market_loss.data[0], d_fake_loss.data[0], g_loss.data[0]))
            else:
                if (step + 1) % self.log_step == 0:
                    print('Step [%d/%d], d_real_loss: %.4f, d_grid_loss: %.4f, d_market_loss: %.4f, '
                          'd_fake_loss: %.4f, g_loss: %.4f, '
                          'dreid_loss: %.4f'
                          % (step + 1, self.train_iters, d_real_loss.data[0], d_grid_loss.data[0],
                             d_market_loss.data[0], d_fake_loss.data[0], g_loss.data[0], dreid_loss.data[0]))

            # save the sampled images
            if (step + 1) % self.sample_step == 0:

                fake_market = self.g12(grid_left)
                fake_grid = self.g21(market_left)

                grid, fake_grid = self.to_data(grid_left), self.to_data(fake_grid)
                market, fake_market = self.to_data(market_left), self.to_data(fake_market)

                merged = self.merge_images(grid, fake_market)
                # path = os.path.join(self.sample_path, 'sample-%d-m-s.png' % (step + 1))
                path = os.path.join(self.sample_path, 'sample-d-g-m.png')
                scipy.misc.imsave(path, merged)
                print('saved %s' % path)

                merged = self.merge_images(market, fake_grid)
                # path = os.path.join(self.sample_path, 'sample-%d-s-m.png' % (step + 1))
                path = os.path.join(self.sample_path, 'sample-d-m-g.png')
                scipy.misc.imsave(path, merged)
                print('saved %s' % path)

            if (step + 1) % 500 == 0:
                # save the model parameters for each epoch
                # g12_path = os.path.join(self.model_path, 'g12-%d.pkl' % (step + 1))
                # g21_path = os.path.join(self.model_path, 'g21-%d.pkl' % (step + 1))
                # d1_path = os.path.join(self.model_path, 'd1-%d.pkl' % (step + 1))
                # d2_path = os.path.join(self.model_path, 'd2-%d.pkl' % (step + 1))
                dreid_path = os.path.join(self.model_path, 'dreid-%d.pkl' % (step + 1))
                # torch.save(self.g12.state_dict(), g12_path)
                # torch.save(self.g21.state_dict(), g21_path)
                # torch.save(self.d1.state_dict(), d1_path)
                # torch.save(self.d2.state_dict(), d2_path)
                torch.save(self.dreid.state_dict(), dreid_path)

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

                if step >= 3000:
                    if (step + 1) % 500 == 0:
                        evaluate_idt(dreid_path)
