import torch
import torch.nn as nn
import numpy as np
from .pos_embed import Pos_Embed

with torch.no_grad():
    # Dijkstra Matrix (D)
    ntu_pos = torch.tensor([[0, 1, 3, 4, 3, 4, 5, 6, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3, 4, 2, 7, 7, 7, 7],
                            [1, 0, 2, 3, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 1, 6, 6, 6, 6],
                            [3, 2, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 1, 6, 6, 6, 6],
                            [4, 3, 1, 0, 3, 4, 5, 6, 3, 4, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 2, 7, 7, 7, 7],
                            [3, 2, 2, 3, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 1, 4, 4, 6, 6],
                            [4, 3, 3, 4, 1, 0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 2, 3, 3, 7, 7],
                            [5, 4, 4, 5, 2, 1, 0, 1, 4, 5, 6, 7, 6, 7, 8, 9, 6, 7, 8, 9, 3, 2, 2, 8, 8],
                            [6, 5, 5, 6, 3, 2, 1, 0, 5, 6, 7, 8, 7, 8, 9, 10, 7, 8, 9, 10, 4, 1, 1, 9, 9],
                            [3, 2, 2, 3, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7, 1, 6, 6, 4, 4],
                            [4, 3, 3, 4, 3, 4, 5, 6, 4, 0, 1, 2, 5, 6, 7, 8, 5, 6, 7, 8, 2, 7, 7, 3, 3],
                            [5, 4, 4, 5, 4, 5, 6, 7, 2, 1, 0, 1, 6, 7, 8, 9, 6, 7, 8, 9, 3, 8, 8, 2, 2],
                            [6, 5, 5, 6, 5, 6, 7, 8, 3, 2, 1, 0, 7, 8, 9, 10, 7, 8, 9, 10, 4, 9, 9, 1, 1],
                            [1, 2, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 0, 1, 2, 3, 2, 3, 4, 5, 3, 8, 8, 8, 8],
                            [2, 3, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 1, 0, 1, 2, 3, 4, 5, 6, 4, 9, 9, 9, 9],
                            [3, 4, 6, 7, 6, 7, 8, 9, 6, 7, 8, 9, 2, 1, 0, 1, 4, 5, 6, 7, 5, 10, 10, 10, 10],
                            [4, 5, 7, 8, 7, 8, 9, 10, 7, 8, 9, 10, 3, 2, 1, 0, 5, 6, 7, 8, 6, 11, 11, 11, 11],
                            [1, 2, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7, 2, 3, 4, 5, 0, 1, 2, 3, 3, 8, 8, 8, 8],
                            [2, 3, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8, 4, 5, 6, 1, 1, 0, 1, 2, 4, 9, 9, 9, 9],
                            [3, 4, 6, 7, 6, 7, 8, 9, 6, 7, 8, 9, 4, 5, 6, 7, 2, 1, 0, 1, 5, 10, 10, 10, 10],
                            [4, 5, 7, 8, 7, 8, 9, 10, 7, 8, 9, 10, 5, 6, 7, 8, 3, 2, 1, 0, 6, 11, 11, 11, 11],
                            [2, 1, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 6, 3, 4, 5, 6, 0, 5, 5, 5, 5],
                            [7, 6, 6, 7, 4, 3, 2, 1, 6, 7, 8, 9, 8, 9, 10, 11, 8, 9, 10, 11, 5, 0, 2, 10, 10],
                            [7, 6, 6, 7, 4, 3, 2, 1, 6, 7, 8, 9, 8, 9, 10, 11, 8, 9, 10, 11, 5, 2, 0, 10, 10],
                            [7, 6, 6, 7, 6, 7, 8, 9, 4, 3, 2, 1, 8, 9, 10, 11, 8, 9, 10, 11, 5, 10, 10, 0, 2],
                            [7, 6, 6, 7, 6, 7, 8, 9, 4, 3, 2, 1, 8, 9, 10, 11, 8, 9, 10, 11, 5, 10, 10, 2, 0]])
    n_ucla_pos = torch.tensor([[0, 1, 2, 3, 3, 4, 5, 6, 3, 4, 5, 6, 1, 2, 3, 4, 1, 2, 3, 4],
                               [1, 0, 1, 2, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5],
                               [2, 1, 0, 1, 1, 2, 3, 4, 1, 2, 3, 4, 3, 4, 5, 6, 3, 4, 5, 6],
                               [3, 2, 1, 0, 2, 3, 4, 5, 2, 3, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7],
                               [3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 4, 5, 6, 7],
                               [4, 3, 2, 3, 1, 0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8, 5, 6, 7, 8],
                               [5, 4, 3, 4, 2, 1, 0, 1, 4, 5, 6, 7, 6, 7, 8, 9, 6, 7, 8, 9],
                               [6, 5, 4, 5, 3, 2, 1, 0, 5, 6, 7, 8, 7, 8, 9, 10, 7, 8, 9, 10],
                               [3, 2, 1, 2, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7],
                               [4, 3, 2, 3, 3, 4, 5, 6, 1, 0, 1, 2, 5, 6, 7, 8, 5, 6, 7, 8],
                               [5, 4, 3, 4, 4, 5, 6, 7, 2, 1, 0, 1, 6, 7, 8, 9, 6, 7, 8, 9],
                               [6, 5, 4, 5, 5, 6, 7, 8, 3, 2, 1, 0, 7, 8, 9, 10, 7, 8, 9, 10],
                               [1, 2, 3, 4, 4, 5, 6, 7, 4, 5, 6, 7, 0, 1, 2, 3, 2, 3, 4, 5],
                               [2, 3, 4, 5, 5, 6, 7, 8, 5, 6, 7, 8, 1, 0, 1, 2, 3, 4, 5, 6],
                               [3, 4, 5, 6, 6, 7, 8, 9, 6, 7, 8, 9, 2, 1, 0, 1, 4, 5, 6, 7],
                               [4, 5, 6, 7, 7, 8, 9, 10, 7, 8, 9, 10, 3, 2, 1, 0, 5, 6, 7, 8],
                               [1, 2, 3, 4, 4, 5, 6, 7, 4, 5, 6, 7, 2, 3, 4, 5, 0, 1, 2, 3],
                               [2, 3, 4, 5, 5, 6, 7, 8, 5, 6, 7, 8, 3, 4, 5, 6, 1, 0, 1, 2],
                               [3, 4, 5, 6, 6, 7, 8, 9, 6, 7, 8, 9, 4, 5, 6, 7, 2, 1, 0, 1],
                               [4, 5, 6, 7, 7, 8, 9, 10, 7, 8, 9, 10, 5, 6, 7, 8, 3, 2, 1, 0]
                               ])
ntu_pos[8][9] = ntu_pos[9][8] = 1
ntu_pos[12][17] = ntu_pos[17][12] = 3
ntu_pos[13][17] = ntu_pos[17][13] = 4
ntu_pos[14][17] = ntu_pos[17][14] = 5
ntu_pos[15][17] = ntu_pos[17][15] = 6


# Spatial_pos = Spatial_pos.cuda()
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)


class Model(nn.Module):
    def __init__(self, len_parts, num_classes, num_joints,
                 num_frames, num_heads, num_persons, num_channels,
                 kernel_size, use_pes=True, config=None,
                 att_drop=0, dropout=0, dropout2d=0, graphw_norm=False, dataset='ntu', koopman='t'):
        super().__init__()
        self.len_parts = len_parts
        in_channels = config[0][0]
        self.out_channels = config[-1][1]
        self.koopman = koopman
        self.num_frames = num_frames // len_parts  # 20
        self.num_joints = num_joints * len_parts  # 150
        self.num_classes = num_classes
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1))

        # self.Spatial_pos_encoder = nn.Conv2d(1, num_heads, 1)
        if dataset == 'ucla':
            with torch.no_grad():
                self.Spatial_pos_weight = torch.zeros(self.num_joints, self.num_joints)  # 150, 150
                for i in range(len_parts):
                    for j in range(len_parts):
                        self.Spatial_pos_weight[i * num_joints:(i + 1) * num_joints,
                        j * num_joints:(j + 1) * num_joints] = torch.add(n_ucla_pos, abs(i - j))
        elif 'ntu' in dataset:
            with torch.no_grad():
                self.Spatial_pos_weight = torch.zeros(self.num_joints, self.num_joints)  # 120, 120
                for i in range(len_parts):
                    for j in range(len_parts):
                        self.Spatial_pos_weight[i * num_joints:(i + 1) * num_joints,
                        j * num_joints:(j + 1) * num_joints] = torch.add(ntu_pos, abs(i - j))
        else:
            raise ValueError()
        self.b = nn.Parameter(torch.ones(self.num_joints, self.num_joints) / num_joints, requires_grad=True)
        if graphw_norm:
            self.Spatial_pos_weight = torch.exp(-self.Spatial_pos_weight / torch.max(self.Spatial_pos_weight))
        else:
            self.Spatial_pos_weight = torch.exp(-self.Spatial_pos_weight)
        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(Graphformer_Block(in_channels, out_channels, qkv_dim,
                                                 num_frames=self.num_frames,
                                                 num_joints=self.num_joints,
                                                 num_heads=num_heads,
                                                 len_parts=len_parts,
                                                 kernel_size=kernel_size,
                                                 use_pes=use_pes,
                                                 att_drop=att_drop
                                                 ))
        self.fc = nn.Linear(self.out_channels, num_classes)
        self.drop_out = nn.Dropout(dropout)
        self.drop_out2d = nn.Dropout2d(dropout2d)
        if koopman == 't' or koopman == 's':
            self.K = nn.Parameter(torch.randn((self.num_classes, 256, 256), requires_grad=True))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)  # 64*2, 3, 120, 25
        self.Spatial_pos_weight = self.Spatial_pos_weight.cuda(x.get_device())  # 150, 150
        # N,3,150,150
        # Spatial_embedding = Spatial_embedding * self.b
        x = x.view(x.size(0), x.size(1), T // self.len_parts, V * self.len_parts)  # 64*2, 3, 20, 150
        x = self.input_map(x)
        att_weight = self.Spatial_pos_weight / 11 + self.b
        for i, block in enumerate(self.blocks):
            x = block(x, att_weight)

        # NM, C, T, V
        if self.koopman != 't' and self.koopman != 's':
            x = x.view(N, M, self.out_channels, -1)
            x = x.permute(0, 1, 3, 2).contiguous().view(N, -1, self.out_channels, 1)
            x = self.drop_out2d(x)
            x = x.mean(3).mean(1)
            x = self.drop_out(x)
            return self.fc(x)
        elif self.koopman == 't':
            c_new = x.size(1)
            x = x.view(N, M, c_new, -1, V)  # 64 * 2 * 256 *t*v
            x = x.mean(-1).mean(1)  # bs * c_new * t_new
        elif self.koopman == 's':
            c_new = x.size(1)
            x = x.view(N, M, c_new, V, -1)  # 64 * 2 * 256 * v*t
            x = x.mean(-1).mean(1)
        x1 = x[:, :, :-1]
        x2 = x[:, :, 1:]
        out = (torch.einsum('cij, bjt -> bcit', self.K, x1) - x2.unsqueeze(1)).norm(dim=-2).mean(dim=-1)
        out = - out
        return out


class Graphformer_Block(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads, len_parts,
                 kernel_size, use_pes=True, att_drop=0):
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)

        # Spatio-Temporal Dijkstra Attention (STDA)
        if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)

        self.len_parts = len_parts
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)),
            nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

        # JTG Sequential Feature Aggregation (like TCN)
        self.tcn = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)),
                                 nn.BatchNorm2d(out_channels))

        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x, atten_weight):
        # atten_weight = atten_weight.cuda(x.get_device())
        N, C, T, V = x.size()
        # Spatio-Temporal Dijkstra Attention (STDA)
        xs = self.pes(x) + x if self.use_pes else x
        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
        attention = self.tan(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        attention = attention * atten_weight.unsqueeze(0).unsqueeze(0) + self.att0s.repeat(N, 1, 1, 1)  # N, 3, 150, 150
        attention = self.drop(attention)
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention]).contiguous().view(N, self.num_heads * self.in_channels, T, V)
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)

        # JTG Sequential Feature Aggregation (TCN)
        xt = self.relu(self.tcn(xs) + self.rest(xs))
        return xt
