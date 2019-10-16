import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class mWDN_RCF(nn.Module):
    def __init__(self, input_size, output_size):
        super(mWDN_RCF, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_2 = input_size // 2
        self.input_size_3 = input_size // 2 // 2

        self.l_filter = [-0.0106, 0.0329, 0.0308, -0.187, -0.028, 0.6309, 0.7148, 0.2304]
        self.h_filter = [-0.2304, 0.7148, -0.6309, -0.028, 0.187, 0.0308, -0.0329, -0.0106]

        self.mWDN1_H = nn.Linear(input_size, input_size)
        self.mWDN1_L = nn.Linear(input_size, input_size)
        self.mWDN2_H = nn.Linear(self.input_size_2, self.input_size_2)
        self.mWDN2_L = nn.Linear(self.input_size_2, self.input_size_2)
        self.mWDN3_H = nn.Linear(self.input_size_3, self.input_size_3)
        self.mWDN3_L = nn.Linear(self.input_size_3, self.input_size_3)

        self.sigmoid = nn.Sigmoid()
        self.a_to_x = nn.AvgPool1d(2)

        self.phi_layer1 = self.make_layer(ResidualBlock, 3, 1)
        self.phi_layer2 = self.make_layer(ResidualBlock, 3, 1)
        self.phi_layer3 = self.make_layer(ResidualBlock, 3, 1)

        self.cmp_mWDN1_H = torch.from_numpy(self.create_W(input_size, False, is_cmp=True)).float()
        self.cmp_mWDN1_L = torch.from_numpy(self.create_W(input_size, True, is_cmp=True)).float()
        self.cmp_mWDN2_H = torch.from_numpy(self.create_W(self.input_size_2, False, is_cmp=True)).float()
        self.cmp_mWDN2_L = torch.from_numpy(self.create_W(self.input_size_2, True, is_cmp=True)).float()
        self.cmp_mWDN3_H = torch.from_numpy(self.create_W(self.input_size_3, False, is_cmp=True)).float()
        self.cmp_mWDN3_L = torch.from_numpy(self.create_W(self.input_size_3, True, is_cmp=True)).float()

        self.mWDN1_H.weight = nn.Parameter(torch.from_numpy(self.create_W(input_size, False)).float(), requires_grad=True)
        self.mWDN1_L.weight = nn.Parameter(torch.from_numpy(self.create_W(input_size, True)).float(), requires_grad=True)
        self.mWDN2_H.weight = nn.Parameter(torch.from_numpy(self.create_W(self.input_size_2, False)).float(), requires_grad=True)
        self.mWDN2_L.weight = nn.Parameter(torch.from_numpy(self.create_W(self.input_size_2, True)).float(), requires_grad=True)
        self.mWDN3_H.weight = nn.Parameter(torch.from_numpy(self.create_W(self.input_size_3, False)).float(), requires_grad=True)
        self.mWDN3_L.weight = nn.Parameter(torch.from_numpy(self.create_W(self.input_size_3, True)).float(), requires_grad=True)

    def make_layer(self, block, num_blocks, inputs, stride=1):
        layers = []
        channels = [4, 16, 16]
        for k in range(num_blocks):
            layers.append(block(inputs, channels[k], stride))
            inputs = channels[k]
        layers.append(nn.Conv1d(channels[-1], 1, kernel_size=1, stride=stride, bias=False))
        layers.append(nn.AdaptiveAvgPool1d(self.output_size))
        return nn.Sequential(*layers)

    def forward(self, input):
        input = input.view(input.size(0), 1, -1)
        # input: torch.Size([32, 1, 720])
        # a1_H: batch_size*seq_len*hidden_size, torch.Size([32, 1, 720])
        a1_H = self.sigmoid(self.mWDN1_H(input))
        a1_L = self.sigmoid(self.mWDN1_L(input))
        # unsqueeze(n)在n维上增加维度，squeeze(n)在n维上消去维度，这里是池化需要增加一个通道的维度
        # x1_H: batch_size*seq_len*(hidden_size / 2)
        x1_H = self.a_to_x(a1_H)
        x1_L = self.a_to_x(a1_L)
        # torch.cat((x1_H, x1_L), 1): batch_size*seq_len*hidden_size
        # u1: batch_size*seq_len*hidden_size
        u1 = self.phi_layer1(torch.cat((x1_H, x1_L), 2))
        c1 = F.softmax(u1, dim=2) #新版本建议加dim

        a2_H = self.sigmoid(self.mWDN2_H(x1_L))
        a2_L = self.sigmoid(self.mWDN2_L(x1_L))
        x2_H = self.a_to_x(a2_H)
        x2_L = self.a_to_x(a2_L)
        u2 = c1 + F.softmax(self.phi_layer2(torch.cat((x2_H, x2_L), 2)), dim=1)
        c2 = F.softmax(u2, dim=2)

        a3_H = self.sigmoid(self.mWDN3_H(x2_L))
        a3_L = self.sigmoid(self.mWDN3_L(x2_L))
        x3_H = self.a_to_x(a3_H)
        x3_L = self.a_to_x(a3_L)
        u3 = c2 + F.softmax(self.phi_layer3(torch.cat((x3_H, x3_L), 2)), dim=1)
        c3 = F.softmax(u3, dim=2)

        return c1, c2, c3

    def create_W(self, P, is_l, is_cmp=False):
        if is_l:
            filter_list = self.l_filter
        else:
            filter_list = self.h_filter
        list_len = len(filter_list)
        max_epsilon = np.random.random_sample()
        while np.abs(max_epsilon) >= np.min(np.abs(filter_list)):
            max_epsilon = np.random.random_sample()
        if is_cmp:
            weight_np = np.zeros((P, P))
        else:
            weight_np = np.random.randn(P, P) * 0.1 * max_epsilon

        for i in range(P):
            filter_index = 0
            for j in range(i, P):
                if filter_index < len(filter_list):
                    weight_np[i][j] = filter_list[filter_index]
                    filter_index += 1
        return weight_np


if __name__ == '__main__':
    net = mWDN_RCF(64, 2)
    print(net)