import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarGCN(nn.Module):
    def __init__(self, input_channels=256, output_channels=256, hidden_channels=512):
        super(SimilarGCN, self).__init__()
        self.conv_g = nn.Conv2d(input_channels, 8, 1, 1)
        self.conv_h = nn.Conv2d(input_channels, 8, 1, 1)
        # self.conv_g2 = nn.Conv2d(8, 1, 1, 1)
        # self.conv_h2 = nn.Conv2d(8, 1, 1, 1)
        self.linear1 = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.LeakyReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(hidden_channels, output_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        N, C, H, W = x.size()
        # nodes_embed1 = self.conv_g(x)
        # nodes_embed2 = self.conv_h(x)
        # # nodes_embed1=self.conv_g2(nodes_embed1).permute(0, 2, 3, 1).contiguous().view(N*H*W, -1)
        # # nodes_embed2=self.conv_h2(nodes_embed2).permute(0, 2, 3, 1).contiguous().view(N*H*W, -1)
        # # adjacency=nodes_embed1+nodes_embed2.t()
        # nodes_embed1=nodes_embed1.permute(0, 2, 3, 1).contiguous().view(N*H*W, -1)
        # nodes_embed2=nodes_embed2.permute(0, 2, 3, 1).contiguous().view(N*H*W, -1)
        # adjacency = nodes_embed1.mm(nodes_embed2.t())
        # # adjacency=F.relu(adjacency)
        # adjacency = F.softmax(adjacency, dim=1)

        # spatial-tempal
        adjacency = torch.zeros(N * H * W, N * H * W).cuda()
        for i in range(N):
            adjacency[i * H * W:(i + 1) * H * W, i * H * W:(i + 1) * H * W] = 1
            if i < N - 1:
                idx1 = range(i * H * W, (i + 1) * H * W)
                idx2 = range((i + 1) * H * W, (i + 2) * H * W)
                adjacency[idx1, idx2] = 1
                adjacency[idx2, idx1] = 1
        adjacency = adjacency+torch.eye(adjacency.size(0)).cuda()
        degree = torch.diag(adjacency.sum(dim=1))
        neg_half_deg = torch.pow(degree, -0.5)
        neg_half_deg[torch.isinf(neg_half_deg)] = 0  # avoid div by 0
        weight = neg_half_deg.mm(adjacency).mm(neg_half_deg)
        nodes = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
        hidden = self.linear1(weight.mm(nodes))
        hidden = self.linear2(weight.mm(hidden))
        # select the max feature
        hidden = hidden.view(N, H, W, -1).permute(0, 3, 1, 2)
        hidden = hidden.max(dim=0, keepdim=True)[0]

        return hidden
