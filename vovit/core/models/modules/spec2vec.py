import torch.nn.functional as F
from einops import rearrange
from torch import nn


class Spec2Vec(nn.Module):
    def __init__(self, last_shape=8):
        super(Spec2Vec, self).__init__()

        # Audio model layers , name of layers as per table 1 given in paper.

        self.conv1 = nn.Conv2d(
            2,
            96,
            kernel_size=(1, 7),
            padding=self.get_padding((1, 7), (1, 1)),
            dilation=(1, 1),
        )

        self.conv2 = nn.Conv2d(
            96,
            96,
            kernel_size=(7, 1),
            padding=self.get_padding((7, 1), (1, 1)),
            dilation=(1, 1),
        )

        self.conv3 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (1, 1)),
            dilation=(1, 1),
        )

        self.conv4 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (2, 1)),
            dilation=(2, 1),
        )

        self.conv5 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (4, 1)),
            dilation=(4, 1),
        )

        self.conv6 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (8, 1)),
            dilation=(8, 1),
        )

        self.conv7 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (16, 1)),
            dilation=(16, 1),
        )

        self.conv8 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (32, 1)),
            dilation=(32, 1),
        )

        self.conv9 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (1, 1)),
            dilation=(1, 1),
        )

        self.conv10 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (2, 2)),
            dilation=(2, 2),
        )

        self.conv11 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (4, 4)),
            dilation=(4, 4),
        )

        self.conv12 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (8, 8)),
            dilation=(8, 8),
        )

        self.conv13 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (16, 16)),
            dilation=(16, 16),
        )

        self.conv14 = nn.Conv2d(
            96,
            96,
            kernel_size=(5, 5),
            padding=self.get_padding((5, 5), (32, 32)),
            dilation=(32, 32),
        )

        self.conv15 = nn.Conv2d(
            96,
            last_shape,
            kernel_size=(1, 1),
            padding=self.get_padding((1, 1), (1, 1)),
            dilation=(1, 1),
        )

        # Batch normalization layers

        self.batch_norm1 = nn.BatchNorm2d(96)
        self.batch_norm2 = nn.BatchNorm2d(96)
        self.batch_norm3 = nn.BatchNorm2d(96)
        self.batch_norm4 = nn.BatchNorm2d(96)
        self.batch_norm5 = nn.BatchNorm2d(96)
        self.batch_norm6 = nn.BatchNorm2d(96)
        self.batch_norm7 = nn.BatchNorm2d(96)
        self.batch_norm8 = nn.BatchNorm2d(96)
        self.batch_norm9 = nn.BatchNorm2d(96)
        self.batch_norm10 = nn.BatchNorm2d(96)
        self.batch_norm11 = nn.BatchNorm2d(96)
        self.batch_norm11 = nn.BatchNorm2d(96)
        self.batch_norm12 = nn.BatchNorm2d(96)
        self.batch_norm13 = nn.BatchNorm2d(96)
        self.batch_norm14 = nn.BatchNorm2d(96)
        self.batch_norm15 = nn.BatchNorm2d(last_shape)

    def get_padding(self, kernel_size, dilation):
        padding = (
            ((dilation[0]) * (kernel_size[0] - 1)) // 2,
            ((dilation[1]) * (kernel_size[1] - 1)) // 2,
        )
        return padding

    def forward(self, input_audio):
        # input audio will be (2,256,256)

        output_layer = F.leaky_relu(self.batch_norm1(self.conv1(input_audio)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm2(self.conv2(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm3(self.conv3(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm4(self.conv4(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm5(self.conv5(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm6(self.conv6(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm7(self.conv7(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm8(self.conv8(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm9(self.conv9(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm10(self.conv10(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm11(self.conv11(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm12(self.conv12(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm13(self.conv13(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm14(self.conv14(output_layer)), negative_slope=0.1)
        output_layer = F.leaky_relu(self.batch_norm15(self.conv15(output_layer)), negative_slope=0.1)

        # output_layer will be (N,8,256,256)
        # we want it to be (N,8*256,256,1)
        output_layer = rearrange(output_layer, 'b c t f -> b c f t ')
        # output_layer = torch.permute(output_layer, [0,1,3,2]).reshape(10,8*256,256)
        output_layer = rearrange(output_layer, 'b c f t -> b (c f) t')
        return output_layer.unsqueeze(-1)  # b (c f) t 1
