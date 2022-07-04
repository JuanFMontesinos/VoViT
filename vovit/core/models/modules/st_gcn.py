import torch
import torch.nn as nn

from .gconv import ConvTemporalGraphical
from .graph import Graph


def init_eiw(x):
    B, T, J = x.shape
    x = x.unsqueeze(2).expand(B, T, J, J)
    x = torch.min(x, x.transpose(2, 3))
    x = x.unsqueeze(2).expand(B, T, 3, J, J)
    return x


class FiLM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bias = nn.Linear(in_channels, out_channels)
        self.scale = nn.Linear(in_channels, out_channels)

    def forward(self, x, c, *args):
        return x * self.scale(c).view(*args) + self.bias(c).view(*args)





class ST_GCN(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence, [X,Y,C]
            :math:`V_{in}` is the number of graph nodes, NUMBER OF JOINTS
            :math:`M_{in}` is the number of instance in a frame. NUMBER OF PEOPLE
    """

    def __init__(self,
                 in_channels,
                 dilated,
                 graph_cfg,
                 temporal_downsample:bool,
                 input_type='x',
                 **kwargs):
        super().__init__()
        self.temporal_downsample = temporal_downsample
        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)
        self.input_type = input_type
        self.dilated = dilated
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 5
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        if kwargs.get('bn_momentum') is not None:
            del kwargs['bn_momentum']
        kwargs['edge_importance_weighting'] = kwargs.get('edge_importance_weighting')
        kwargs['A'] = A
        kwargs['dilation'] = 2 if dilated else 1
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 32,
                         kernel_size, 1,
                         residual=False, **kwargs0),
            st_gcn_block(32, 32, kernel_size, 1, **kwargs),
            st_gcn_block(32, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 128, kernel_size, 1, **kwargs),
            st_gcn_block(128, 256, kernel_size, 1, **kwargs),
            st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        ))

        if self.temporal_downsample:
            self.st_gcn_networks = nn.ModuleList((
                st_gcn_block(in_channels, 32,
                             kernel_size, 1,
                             residual=False, **kwargs0),
                st_gcn_block(32, 32, kernel_size, 1, **kwargs),
                st_gcn_block(32, 64, kernel_size, 2, **kwargs),
                st_gcn_block(64, 64, kernel_size, 1, **kwargs),
                st_gcn_block(64, 128, kernel_size, 2, **kwargs),
                st_gcn_block(128, 128, kernel_size, 1, **kwargs),
                st_gcn_block(128, 256, kernel_size, 2, **kwargs),
                st_gcn_block(256, 256, kernel_size, 1, **kwargs),
            ))

    def forward(self, x, *args):
        args = list(args)
        if x.shape[1] == 3:
            args.append(x[:, 2, ...])
        x = self.extract_feature(x, *args)

        return x

    def extract_feature(self, x, *args):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad

        for gcn in self.st_gcn_networks:
            x, _ = gcn(x, self.A, *args)
        return x



class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 dilation=1,
                 edge_importance_weighting='static',
                 A=None,
                 activation='relu'):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = (dilation * (kernel_size[0] - 1)) // 2
        padding = (padding, 0)

        self.ctype = edge_importance_weighting
        self.activation = activation
        if edge_importance_weighting == 'static':
            self.edge_importance = 1.
            self.edge_importance_weighting = True
        elif edge_importance_weighting == 'dynamic':
            self.edge_importance_weighting = True
            self.edge_importance = nn.Parameter(torch.ones(A.shape))

        else:
            raise ValueError('edge_importance_weighting (%s) not implemented')

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
                dilation=(dilation, 1),
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        if self.edge_importance_weighting:
            A = A * self.edge_importance
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        if self.activation == 'relu':
            return self.relu(x), A
        else:
            return x, A
