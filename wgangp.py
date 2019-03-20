import imageio
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch import autograd
from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

class WGANGP():
    def __init__(self, generator, discriminator, g_optmizer, d_optimizer,
                 latent_shape, dataset_name, n_critic=5, gamma=10,
                 save_every=20, use_cuda=True, logdir=None):

        self.G = generator
        self.D = discriminator
        self.G_opt = g_optmizer
        self.D_opt = d_optimizer
        self.latent_shape = latent_shape
        self.dataset_name = dataset_name
        self.n_critic = n_critic
        self.gamma = gamma
        self.save_every = save_every
        self.use_cuda = use_cuda
        self.writer = SummaryWriter(logdir)
        self.steps = 0
        self._fixed_z = torch.randn(64, *latent_shape)
        self.hist = []
        self.images = []

        if self.use_cuda:
            self._fixed_z = self._fixed_z.cuda()
            self.G.cuda()
            self.D.cuda()

    def train(self, data_loader, n_epochs):
        self._save_gif()
        for epoch in range(1, n_epochs + 1):
            print('Starting epoch {}...'.format(epoch))
            self._train_epoch(data_loader)

            if epoch % self.save_every == 0 or epoch == n_epochs:
                torch.save(self.G.state_dict(), self.dataset_name + '_gen_{}.pt'.format(epoch))
                torch.save(self.D.state_dict(), self.dataset_name + '_disc_{}.pt'.format(epoch))

    def _train_epoch(self, data_loader):
        for i, (data, _) in enumerate(data_loader):
            self.steps += 1
            data = Variable(data)
            if self.use_cuda:
                data = data.cuda()

            d_loss, grad_penalty = self._discriminator_train_step(data)
            self.writer.add_scalars('losses', {'d_loss': d_loss, 'grad_penalty': grad_penalty}, self.steps)
            self.hist.append({'d_loss': d_loss, 'grad_penalty': grad_penalty})

            if i % 200 == 0:
                img_grid = make_grid(self.G(self._fixed_z).cpu().data, normalize=True)
                self.writer.add_image('images', img_grid, self.steps)

            if self.steps % self.n_critic == 0:
                g_loss = self._generator_train_step(data.size(0))
                self.writer.add_scalars('losses', {'g_loss': g_loss}, self.steps)
                self.hist[-1]['g_loss'] = g_loss

        print('    g_loss: {:.3f} d_loss: {:.3f} grad_penalty: {:.3f}'.format(g_loss, d_loss, grad_penalty))

    def _discriminator_train_step(self, data):
        batch_size = data.size(0)
        generated_data = self._sample(batch_size)
        grad_penalty = self._gradient_penalty(data, generated_data)
        d_loss = self.D(generated_data).mean() - self.D(data).mean() + grad_penalty
        self.D_opt.zero_grad()
        d_loss.backward()
        self.D_opt.step()
        return d_loss.item(), grad_penalty.item()

    def _generator_train_step(self, batch_size):
        self.G_opt.zero_grad()
        generated_data = self._sample(batch_size)
        g_loss = -self.D(generated_data).mean()
        g_loss.backward()
        self.G_opt.step()
        return g_loss.item()

    def _gradient_penalty(self, data, generated_data, gamma=10):
        batch_size = data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1)
        epsilon = epsilon.expand_as(data)


        if self.use_cuda:
            epsilon = epsilon.cuda()

        interpolation = epsilon * data.data + (1 - epsilon) * generated_data.data
        interpolation = Variable(interpolation, requires_grad=True)

        if self.use_cuda:
            interpolation = interpolation.cuda()

        interpolation_logits = self.D(interpolation)
        grad_outputs = torch.ones(interpolation_logits.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=interpolation_logits,
                                  inputs=interpolation,
                                  grad_outputs=grad_outputs,
                                  create_graph=True,
                                  retain_graph=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        return self.gamma * ((gradients_norm - 1) ** 2).mean()

    def _sample(self, n_samples):
        z = Variable(torch.randn(n_samples, *self.latent_shape))
        if self.use_cuda:
            z = z.cuda()
        return self.G(z)

    def _save_gif(self):
        grid = make_grid(self.G(self._fixed_z).cpu().data, normalize=True)
        grid = np.transpose(grid.numpy(), (1, 2, 0))
        self.images.append(grid)
        imageio.mimsave('{}.gif'.format(self.dataset_name), self.images)
