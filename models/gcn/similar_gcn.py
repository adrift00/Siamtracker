import torch
import torch.nn as nn


class SimilarGCN(nn.Module):
    def __init__(self,input_channels=256,output_channels=256):
        super(SimilarGCN,self).__init__()
        self.conv_g=nn.Conv2d(input_channels,8,1,1)
        self.conv_h=nn.Conv2d(input_channels,8,1,1)
        hidden_channels=512
        self.linear1=nn.Linear(input_channels,hidden_channels)
        self.linear2=nn.Linear(hidden_channels,output_channels)
    
    def forward(self,x):
        N,C,H,W=x.size()
        nodes_embed1=self.conv_g(x)
        nodes_embed2=self.conv_h(x)
        nodes_embed1=nodes_embed1.permute(0,2,3,1).contiguous().view(N*H*W,-1)
        nodes_embed2=nodes_embed2.permute(0,2,3,1).contiguous().view(N*H*W,-1)
        adjacency=nodes_embed1.mm(nodes_embed2.t())
        adjacency=adjacency+torch.eye(adjacency.size(0)).cuda()
        adjacency=torch.softmax(adjacency,dim=1)
        degree=torch.diag(adjacency.sum(dim=1))
        degree=degree+1e-7
        neg_half_deg=torch.pow(degree,-0.5)
        weight=neg_half_deg.mm(adjacency).mm(neg_half_deg)
        nodes=x.permute(0,2,3,1).contiguous().view(-1,C)
        hidden=self.linear1(weight.mm(nodes))
        hidden=self.linear2(weight.mm(hidden))
        # select the max feature
        hidden=hidden.view(N,H,W,-1).permute(0,3,1,2)
        #import ipdb;ipdb.set_trace()
        hidden=hidden.max(dim=0,keepdim=True)[0]
        
        return hidden
