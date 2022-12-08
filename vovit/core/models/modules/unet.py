from typing import List

import torch
import torch.nn as nn
from numbers import Number


def isnumber(x):
    return isinstance(x, Number)


class FiLM_parent(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bias = nn.Linear(in_channels, out_channels)
        self.scale = nn.Linear(in_channels, out_channels)

    def forward(self, x, c):
        scale, bias = self.scale(c), self.bias(c)
        return scale, bias


class FiLM(FiLM_parent):
    def forward(self, x, c):
        scale = self.scale(c).unsqueeze(2).unsqueeze(2)
        bias = self.bias(c).unsqueeze(2).unsqueeze(2)
        return x * scale + bias


def crop(img, i, j, h, w):
    """Crop the given Image.
    Args:
        img Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img[:, :, i:i + h, j:j + w]


def center_crop(img, output_size):
    """This function is prepared to crop tensors provided by dataloader.
    Cited tensors has shape [1,N_maps,H,W]
    """
    _, _, h, w = img.size()
    th, tw = output_size[0], output_size[1]
    i = int(torch.round((h - th) / torch.tensor(2.)))
    j = int(torch.round((w - tw) / torch.tensor(2.)))
    return crop(img, i, j, th, tw)


def conv_bn_lr_do(dim_in, dim_out, kernel_size, stride, padding, bias, use_conv, use_bn, relu, dropout, **kwargs):
    layers = []
    if use_conv:
        layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias))
    if use_bn:
        layers.append(nn.BatchNorm2d(dim_out, **kwargs))
    if isnumber(relu):
        if relu >= 0:
            layers.append(nn.LeakyReLU(relu, inplace=True))
    if dropout:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


class ConvolutionalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, film, kernel_conv=3, kernel_MP=2, stride_conv=1, stride_MP=2, padding=1,
                 bias=True, dropout=False, useBN=False, bn_momentum=0.1,
                 mode='upconv', architecture='original', **kwargs):
        super(ConvolutionalBlock, self).__init__()
        """Defines a (down)convolutional  block
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not
            useBN: Use batch normalization

        Forward:
            Returns:
                to_cat: output previous to Max Pooling for skip connections
                to_down: Max Pooling output to be used as input for next block
        """
        assert isinstance(dropout, Number)
        assert mode in ['upconv', 'upsample']
        assert architecture in ['original', 'sop']
        if useBN:
            bias = False
        self.film = isnumber(film)
        self.useBN = useBN
        # HARCODING DROPOUT IN ENCODER AS FALSE
        self.dropout = 0
        self.opt_layer = None

        if isnumber(film):
            self.film_layer = FiLM(film, dim_out)

        # Layer 1
        self.layer = conv_bn_lr_do(dim_in, dim_out, kernel_conv, stride_conv, padding,
                                   bias, True, self.useBN, relu=-1, dropout=False, momentum=bn_momentum)
        self.ReLu = nn.LeakyReLU(0.1, inplace=True)
        self.MaxPooling = nn.MaxPool2d(kernel_size=kernel_MP, stride=stride_MP, padding=0, dilation=1,
                                       return_indices=False, ceil_mode=False)
        if architecture == 'original':
            # Layer 2
            self.opt_layer = conv_bn_lr_do(dim_out, dim_out, kernel_conv, stride_conv, padding,
                                           bias, True, self.useBN, relu=0.1, dropout=False, momentum=bn_momentum)

    def forward(self, *args):
        if self.film:
            x, c = args
        else:
            x = args[0]

        x = self.layer(x)
        if self.film:
            x = self.film_layer(x, c)
        x = self.ReLu(x)
        if self.opt_layer is not None:
            x = self.opt_layer(x)
        to_cat = x
        to_down = self.MaxPooling(to_cat)

        return to_cat, to_down


class TransitionBlock(nn.Module):
    """Specific class for lowest block. Change values carefully.
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not
            useBN: Use batch normalization
        Forward:
            Input:
                x: previous block input.
            Returns:
                x: block output
    """

    def __init__(self, dim_in, dim_out, film, kernel_conv=3, kernel_UP=2, stride_conv=1, stride_UP=2, padding=1,
                 bias=True, useBN=False, verbose=False, bn_momentum=0.1, mode='upconv', architecture='original',
                 finalblock=None,  # Placeholder
                 **kwargs):
        super(TransitionBlock, self).__init__()
        # LAYERS AREN'T DEFINED IN ORDER
        self.useBN = useBN
        self.mode = mode
        self.verbose = verbose
        self.opt_layer = None
        self.film = isnumber(film)
        if useBN:
            bias = False
        assert mode in ['upconv', 'upsample']
        assert architecture in ['original', 'sop']
        dropout = 0
        if architecture == 'original' or mode == 'upconv':
            dim_inner = dim_in
        elif architecture == 'sop' and mode == 'upsample':
            dim_inner = dim_out

        self.layer = conv_bn_lr_do(dim_out, dim_inner, kernel_conv, stride_conv, padding,
                                   bias, True, useBN, relu=-1, dropout=dropout, momentum=bn_momentum)

        if architecture == 'original':
            # Layer 2
            diml2 = dim_in * (mode == 'upconv') + dim_out * (mode == 'upsample')
            self.opt_layer = conv_bn_lr_do(dim_inner, diml2, kernel_conv, stride_conv, padding,
                                           bias, True, useBN, relu=0, dropout=dropout, momentum=bn_momentum)

        self.ReLu = nn.ReLU(inplace=True)

        if mode == 'upconv':
            self.AtrousConv = nn.ConvTranspose2d(dim_inner, dim_out, kernel_size=kernel_UP, stride=stride_UP,
                                                 padding=0, dilation=1)
        elif mode == 'upsample':
            self.AtrousConv = nn.Upsample(scale_factor=kernel_UP, mode='bilinear', align_corners=True)
        if self.film:
            self.film_layer = FiLM(film, dim_inner)

    def forward(self, *args):
        if self.film:
            x, c = args
        else:
            x = args[0]
        x = self.layer(x)
        if self.verbose:
            print('Latent space shape:%s' % str(x.shape))
        if self.film:
            x = self.film_layer(x, c)
        x = self.ReLu(x)  # This is the latent space
        if self.opt_layer is not None:
            x = self.opt_layer(x)
        to_up = self.AtrousConv(x)

        return to_up


class AtrousBlock(nn.Module):
    def __init__(self, dim_in, dim_out, film, kernel_conv=3, kernel_UP=2, stride_conv=1, stride_UP=2, padding=1,
                 bias=True, useBN=False, finalblock=False, verbose=False, bn_momentum=0.1, dropout=False,
                 mode='upconv', architecture='original', **kwargs):
        """Defines a upconvolutional  block
        Args:
            dim_in: int dimension of feature maps of block input.
            dim_out: int dimension of feature maps of block output.
            kernel_conv: int or tuple kernel size for convolutions
            kernel_MP: int or tuple kernel size for Max Pooling
            stride_conv: int or tuple stride for convolutions
            stride_MP: int or tuple stride for Max Pooling
            padding: padding for convolutions
            bias: bool Set bias or not
            useBN: Use batch normalization
            finalblock: bool Set true if it's the last upconv block not to do upconvolution.
            mode: 'upconv' to use Atrous Convolutions or 'upsample' to use linear interp
        Forward:
            Input:
                x: previous block input.
                to_cat: skip connection input.
            Returns:
                x: block output
        """
        super(AtrousBlock, self).__init__()
        self.opt_layer = None
        self.finalblock = finalblock
        self.verbose = verbose
        self.film = isnumber(film)
        if useBN:
            bias = False
        assert isinstance(dropout, Number)
        assert mode in ['upconv', 'upsample']
        assert architecture in ['original', 'sop']
        if self.finalblock:  # Prevents dropout in the last block as it would set to zero outgoing values
            dropout = False
        if architecture == 'sop' and mode == 'upsample' and not finalblock:
            dim_inner = dim_out
        else:
            dim_inner = dim_in

        self.layer = conv_bn_lr_do(2 * dim_in, dim_inner, kernel_conv, stride_conv, padding,
                                   bias, True, useBN, relu=-1, dropout=dropout, momentum=bn_momentum)

        if architecture == 'original':
            # Layer 2
            diml2 = dim_in * ((mode == 'upconv' or finalblock)) + dim_out * (mode == 'upsample') * (not finalblock)
            self.opt_layer = conv_bn_lr_do(dim_inner, diml2, kernel_conv, stride_conv, padding,
                                           bias, True, useBN, relu=0, dropout=dropout, momentum=bn_momentum)

        self.ReLu = nn.ReLU(inplace=True)
        if not finalblock:
            if mode == 'upconv':
                self.AtrousConv = nn.ConvTranspose2d(dim_inner, dim_out, kernel_size=kernel_UP, stride=stride_UP,
                                                     padding=0, dilation=1)
            elif mode == 'upsample':
                self.AtrousConv = nn.Upsample(scale_factor=kernel_UP, mode='bilinear', align_corners=True)
        if self.film:
            self.film_layer = FiLM(film, dim_inner)

    def forward(self, *args):
        if self.film:
            x, to_cat, c = args
        else:
            x, to_cat = args[:2]

        if self.verbose:
            print('Incoming variable from previous Upconv Block: {}'.format(x.size()))

        to_cat = center_crop(to_cat, x.size()[2:4])
        x = torch.cat((x, to_cat), dim=1)
        x = self.layer(x)
        if self.film:
            x = self.film_layer(x, c)
        x = self.ReLu(x)
        if self.opt_layer is not None:
            x = self.opt_layer(x)

        if not self.finalblock:
            x = self.AtrousConv(x)
        return x


class UNet(nn.Module):
    """It's recommended to be very careful  while managing vectors, since they are inverted to
    set top blocks as block 0. Notice there are N upconv blocks and N-1 downconv blocks as bottom block
    is considered as upconvblock.

    C-U-Net based on this paper https://arxiv.org/pdf/1904.05979.pdf
    """
    """
    Example:
    model = UNet([64,128,256,512,1024,2048,4096],K,useBN=True,input_channels=1)

        K(int) : Amount of outgoing channels 
        useBN (bool): Whether to use or not batch normalization
        input_channels (int): Amount of input channels
        dimension_vector (tuple/list): Its length defines amount of block. Elements define amount of filters per block



    """

    def __init__(self, layer_channels: List[int], output_channels: int, film, verbose: bool = False,
                 layer_kernels: str = None, useBN=False, input_channels=1, activation=None, **kwargs):
        super(UNet, self).__init__()
        self.K = output_channels
        self.printing = verbose
        if film is None:
            self.film = None
            self.film_where = []
        elif isinstance(film, (tuple, list)):
            self.film, self.film_where = film
        else:
            raise TypeError('Argument film must be None or tuple/list')
        self.useBN = useBN
        self.input_channels = input_channels
        self.dim = layer_channels
        self._set_layer_kernels(layer_kernels)
        self.init_assertion(**kwargs)

        self.vec = range(len(self.dim))
        if 'e' in self.film_where:
            self.encoder = self.add_encoder(input_channels, self.film, **kwargs)
        else:
            self.encoder = self.add_encoder(input_channels, None, **kwargs)
        if 'd' in self.film_where:
            film_d = self.film
        else:
            film_d = None
        if 'l' in self.film_where:
            film_l = self.film
        else:
            film_l = None

        self.decoder = self.add_decoder(film_d, film_l, **kwargs)

        self.activation = activation
        self.final_conv = nn.Conv2d(self.dim[0], self.K, kernel_size=1, stride=1, padding=0)
        if self.activation is not None:
            self.final_act = self.activation

    def _set_layer_kernels(self, layer_kernels: str):
        assert hasattr(self, 'dim'), '# Layer channels not defined'
        assert layer_kernels is None or isinstance(layer_kernels, str), f'layer_kernels must be {None} or {str}'
        if layer_kernels is None:
            n_elements = len(self.dim) - 1
            self.kernels = ''
            for _ in range(n_elements):
                self.kernels += 's'
        else:
            layer_kernels = layer_kernels.lower()
            for element in layer_kernels:
                assert element in ['s', 'f', 't'], f'Allowed characters are t,s and f but {element} found'
            self.kernels = layer_kernels
        assert len(self.kernels) == len(
            self.dim) - 1, f'layer_kernels must have N-1 elements as the inner block is always squared'

    def init_assertion(self, **kwargs):
        if self.film is not None:
            assert isinstance(self.film, Number)
            for c in list(self.film_where):
                assert c.lower() in ('e', 'd', 'l')
        assert isinstance(self.dim, (tuple, list))
        for x in self.dim:
            assert x % 2 == 0

        assert isinstance(self.input_channels, int)
        assert self.input_channels > 0
        assert isinstance(self.K, int)
        assert self.K > 0
        if kwargs.get('dropout') is not None:
            dropout = kwargs['dropout']
            assert isinstance(dropout, Number)
            assert dropout >= 0
            assert dropout <= 1
        if kwargs.get('bn_momentum') is not None:
            bn_momentum = kwargs['bn_momentum']
            assert isinstance(bn_momentum, Number)
            assert bn_momentum >= 0
            assert bn_momentum <= 1
        if isnumber(self.film) and self.useBN == False:
            raise ValueError(
                'Conditioned U-Net enabled but batch normalization disabled. C-UNet only available with BN on.'
                ' Note: from a Python perspective, booleans are integers, thus numbers')

    def add_encoder(self, input_channels, film, **kwargs):
        encoder = []
        if kwargs.get('architecture') == 'sop':
            kwargs['bias'] = False
            kwargs['kernel_conv'] = 5
            kwargs['padding'] = 2
        for i in range(len(self.dim) - 1):  # There are len(self.dim)-1 downconv blocks
            if self.printing:
                print('Building Downconvolutional Block {} ...OK'.format(i))
            block = ConvolutionalBlock(dim_in=input_channels if i == 0 else self.dim[i - 1],
                                       dim_out=self.dim[i], film=film, useBN=self.useBN,
                                       kernel_MP=self._calc_mp_kernel(self.kernels[i]),
                                       stride_MP=self._calc_mp_kernel(self.kernels[i]),
                                       **kwargs)
            encoder.append(block)
        encoder = nn.Sequential(*encoder)
        return encoder

    def _calc_mp_kernel(self, pooling):
        if pooling == 's':
            return 2
        elif pooling == 't':
            return tuple([2, 1])
        elif pooling == 'f':
            return tuple([1, 2])
        else:
            raise NotImplementedError

    def add_decoder(self, film_d, film_l, **kwargs):
        if kwargs.get('architecture') == 'sop':
            kwargs['bias'] = False
        decoder = []
        kernels = 's' + self.kernels
        for i in self.vec[::-1]:  # [::-1] inverts the order to set top layer as layer 0 and to order
            # layers from the bottom to above according to  flow of information.
            if self.printing:
                print('Building Upconvolutional Block {}...OK'.format(i))

            constructor = TransitionBlock if i == max(self.vec) else AtrousBlock

            block = constructor(dim_in=self.dim[i], dim_out=self.dim[i - 1],
                                film=film_l if i == max(self.vec) else film_d,
                                verbose=self.printing,
                                useBN=self.useBN,
                                finalblock=True if i == 0 else False,
                                kernel_UP=self._calc_mp_kernel(kernels[i]),
                                stride_UP=self._calc_mp_kernel(kernels[i]),
                                **kwargs)
            decoder.append(block)
        decoder = nn.Sequential(*decoder)
        return decoder

    def forward(self, *args):
        if isnumber(self.film):
            x, c = args
        else:
            x = args[0]
        if self.printing:
            print('UNet input size {0}'.format(x.size()))
        to_cat_vector = []
        for i in range(len(self.dim) - 1):
            if self.printing:
                print('Forward Prop through DownConv block {}'.format(i))
            if isnumber(self.film):

                to_cat, x = self.encoder[i](x, c)
            else:
                to_cat, x = self.encoder[i](x)
            to_cat_vector.append(to_cat)
        for i in self.vec:
            if self.printing:
                print('Concatenating and Building  UpConv Block {}'.format(i))
            if i == 0:
                if isnumber(self.film):
                    x = self.decoder[i](x, c)
                else:
                    x = self.decoder[i](x)
            else:
                if isnumber(self.film):
                    x = self.decoder[i](x, to_cat_vector[-i], c)
                else:
                    x = self.decoder[i](x, to_cat_vector[-i])
        x = self.final_conv(x)
        if self.activation is not None:
            x = self.final_act(x)
        if self.printing:
            print('UNet Output size {}'.format(x.size()))

        return x
