import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class DilatedCausalConv1d(nn.Module):
    """Dilated Causal Convolution for WaveNet"""

    def __init__(self, channels, dilation=1):
        super(DilatedCausalConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(channels, channels,
                                    kernel_size=2, stride=1,  # Fixed for WaveNet
                                    dilation=dilation,
                                    padding=0,  # Fixed for WaveNet dilation
                                    bias=False)  # Fixed for WaveNet but not sure

    def forward(self, x):
        output = self.conv(x)

        return output


class CausalConv1d(nn.Module):
    """Causal Convolution for WaveNet"""

    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()

        # padding=1 for same size(length) between input and output for causal convolution
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=2, stride=1, padding=1,
                                    bias=False)  # Fixed for WaveNet but not sure

    def forward(self, x):
        output = self.conv(x)

        # remove last value for causal convolution
        return output[:, :, :-1]


class ResidualBlock(torch.nn.Module):
    def __init__(self, res_channels, skip_channels, dilation):
        """
        Residual block
        :param res_channels: number of residual channel for input, output
        :param skip_channels: number of skip channel for output
        :param dilation:
        """
        super(ResidualBlock, self).__init__()

        self.dilated = DilatedCausalConv1d(res_channels, dilation=dilation)
        self.conv_res = torch.nn.Conv1d(res_channels, res_channels, 1)
        self.conv_skip = torch.nn.Conv1d(res_channels, skip_channels, 1)

        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        """
        :param x:
        :return:
        """
        output = self.dilated(input)

        # PixelCNN gate
        gated_tanh = self.gate_tanh(output)
        gated_sigmoid = self.gate_sigmoid(output)
        gated = gated_tanh * gated_sigmoid

        # Residual network
        output = self.conv_res(gated)
        input_cut = input[:, :, -output.size(2):]
        output += input_cut
        skip_out = self.conv_skip(gated)

        return output, skip_out


class WaveNet(nn.Module):
    def __init__(self, in_depth=256, res_channels=32, skip_channels=512, dilation_depth=8, n_repeat=5):
        """
        input: Tensor[batch, channel, length]
        Args:
            in_depth:
            res_channels:
            skip_channels:
            dilation_depth:
            n_repeat:
        """
        super(WaveNet, self).__init__()
        self.dilations = [2 ** i for i in range(dilation_depth)] * n_repeat
        self.main = nn.ModuleList([ResidualBlock(res_channels, skip_channels, dilation) for dilation in self.dilations])
        # self.pre = nn.Embedding(in_depth, res_channels)
        self.pre_conv = CausalConv1d(in_channels=in_depth, out_channels=res_channels)
        self.post = nn.Sequential(nn.ReLU(),
                                  nn.Conv1d(skip_channels, skip_channels, 1),
                                  nn.ReLU(),
                                  nn.Conv1d(skip_channels, in_depth, 1))

    def forward(self, inputs):
        outputs = self.preprocess(inputs)
        skip_connections = []

        for layer in self.main:
            outputs, skip = layer(outputs)
            skip_connections.append(skip)

        outputs = sum([s[:, :, -outputs.size(2):] for s in skip_connections])
        outputs = self.post(outputs)
        return outputs

    def preprocess(self, inputs):
        out = self.pre_conv(inputs)
        # out = out.transpose(1, 2)
        # out = self.pre_conv(out)
        return out

class GenLSTM(nn.Module):

    def __init__(self, in_depth=256, length=5, stride=500):
        '''
        input: Tensor[batch, time_step, input_size]
        Args:
            in_depth:
            length:
            stride:
        '''

        super(GenLSTM, self).__init__()
        self.length = length
        self.stride = stride
        self.main = nn.LSTM(in_depth, in_depth, batch_first=True)

    def preprocess(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        out = inputs[:, 0:-1:self.stride, :]
        return out

    def forward(self, inputs):
        output, h = self.main(inputs)
        return output

class DisLSTM(nn.Module):
    def __init__(self, in_depth=256):
        super(DisLSTM, self).__init__()
        self.main = nn.LSTM(in_depth, in_depth, batch_first=True)
        self.out = nn.Sequential(nn.Linear(in_depth, 1),
                                 nn.Sigmoid())


    def forward(self, inputs):
        r_out, _ = self.main(inputs)
        output = self.out(r_out[:, -1, :])
        return output