import datetime
import os
import time
import timeit
import numpy as np
import torch as th


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    def __init__(self, depth=7, latent_size=512, dilation=1, use_spectral_norm=True):
 
        from torch.nn import ModuleList, Conv2d
        from MSG_GAN.CustomLayers import GenGeneralConvBlock, GenInitialBlock

        super().__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.depth = depth
        self.latent_size = latent_size
        self.spectral_norm_mode = None
        self.dilation = dilation

        # register the modules required for the GAN Below ...
        # create the ToRGB layers for various outputs:
        def to_rgb(in_channels):
            return Conv2d(in_channels, 3, (1, 1), bias=True)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([GenInitialBlock(self.latent_size)])
        self.rgb_converters = ModuleList([to_rgb(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = GenGeneralConvBlock(self.latent_size, self.latent_size,
                                            dilation=dilation)
                rgb = to_rgb(self.latent_size)
            else:
                layer = GenGeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    dilation=dilation
                )
                rgb = to_rgb(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()

    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = True

    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv_1)
            remove_spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = False

    def forward(self, x):
        """
        forward pass of the Generator
        :param x: input noise
        :return: *y => output of the generator at various scales
        """
        from torch import tanh
        outputs = []  # initialize to empty list

        y = x  # start the computational pipeline
        for block, converter in zip(self.layers, self.rgb_converters):
            y = block(y)
            outputs.append(tanh(converter(y)))

        return outputs


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    def __init__(self, depth=7, feature_size=512, dilation=1, use_spectral_norm=True):

        from torch.nn import ModuleList
        from MSG_GAN.CustomLayers import DisGeneralConvBlock, DisFinalBlock
        from torch.nn import Conv2d

        super().__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert feature_size >= np.power(2, depth - 4), \
                "feature size cannot be produced"

        # create state of the object
        self.depth = depth
        self.feature_size = feature_size
        self.spectral_norm_mode = None
        self.dilation = dilation

        # create the fromRGB layers for various inputs:
        def from_rgb(out_channels):
            return Conv2d(3, out_channels, (1, 1), bias=True)

        self.rgb_to_features = ModuleList([from_rgb(self.feature_size // 2)])

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([DisFinalBlock(self.feature_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i > 2:
                layer = DisGeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 2)),
                    dilation=dilation
                )
                rgb = from_rgb(int(self.feature_size // np.power(2, i - 1)))
            else:
                layer = DisGeneralConvBlock(self.feature_size, self.feature_size // 2,
                                            dilation=dilation)
                rgb = from_rgb(self.feature_size // 2)
        
                
            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # just replace the last converter
        self.rgb_to_features[self.depth - 1] = \
            from_rgb(self.feature_size // np.power(2, i - 2))

        # if spectral normalization is on:
        if use_spectral_norm:
            self.turn_on_spectral_norm()

    def turn_on_spectral_norm(self):
        """
        private helper for turning on the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is False, \
                "can't apply spectral_norm. It is already applied"

        # apply the same to the remaining relevant blocks
        for module in self.layers:
            module.conv_1 = spectral_norm(module.conv_1)
            module.conv_2 = spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = True

    def turn_off_spectral_norm(self):
        """
        private helper for turning off the spectral normalization
        :return: None (has side effect)
        """
        from torch.nn.utils import remove_spectral_norm

        if self.spectral_norm_mode is not None:
            assert self.spectral_norm_mode is True, \
                "can't remove spectral_norm. It is not applied"

        # remove the applied spectral norm
        for module in self.layers:
            remove_spectral_norm(module.conv_1)
            remove_spectral_norm(module.conv_2)

        # toggle the state variable:
        self.spectral_norm_mode = False

    def forward(self, inputs):
      
        from torch.nn import AvgPool2d, LeakyReLU

        assert len(inputs) == self.depth, \
            "Mismatch between input and Network scales"

        y = self.rgb_to_features[self.depth - 1](inputs[self.depth - 1]) #32x256x256
        y = self.layers[self.depth - 1](y)
        #y = th.cat((inputs[self.depth - 1], y), dim=1)
        #y = self.layers[self.depth - 1](y) #alpha 64x128x128
        #y_ = AvgPool2d(2)(inputs[self.depth - 1]) #3x128x128
        #y_ = self.rgb_to_features[self.depth - 2](y_) #64x128x128
        #y_ = LeakyReLU(0.2)(y_)
        #y = y * 0.95 + y_ * 0.05
        for x, block, converter in \
                zip(reversed(inputs[:-1]),
                    reversed(self.layers[:-1]),
                    reversed(self.rgb_to_features[:-1])):
            input_part = converter(x)  # convert the input:
            y = th.cat((input_part, y), dim=1)  # concatenate the inputs:
            y = block(y)  # apply the block
        return y


class MSG_GAN:

    def __init__(self, depth=7, latent_size=512, gen_dilation=1,
                 dis_dilation=1, use_spectral_norm=True, device=th.device("cpu")):
        """ constructor for the class """
        from torch.nn import DataParallel

        self.gen = Generator(depth, latent_size, dilation=gen_dilation,
                             use_spectral_norm=use_spectral_norm).to(device)
        self.dis = Discriminator(depth, latent_size, dilation=dis_dilation,
                                 use_spectral_norm=use_spectral_norm).to(device)

        # Create the Generator and the Discriminator
        if device == th.device("cuda"):
            self.gen = DataParallel(self.gen)
            self.dis = DataParallel(self.dis)

        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.device = device

        # by default the generator and discriminator are in eval mode
        self.gen.eval()
        self.dis.eval()

    def optimize_discriminator(self, dis_optim, noise, real_batch, loss_fn):

        # generate a batch of samples
        fake_samples = self.gen(noise)
        fake_samples = list(map(lambda x: x.detach(), fake_samples))

        loss = loss_fn.dis_loss(real_batch, fake_samples)

        # optimize discriminator
        dis_optim.zero_grad()
        loss.backward(retain_graph=True)
        dis_optim.step()

        return loss.item()

    def optimize_generator(self, gen_optim, noise, real_batch, loss_fn):

        # generate a batch of samples
        fake_samples = self.gen(noise)

        loss = loss_fn.gen_loss(real_batch, fake_samples)

        # optimize discriminator
        gen_optim.zero_grad()
        loss.backward(retain_graph=True)
        gen_optim.step()

        return loss.item()