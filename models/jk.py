import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge


class JK(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(JK, self).__init__()
        self.body = JK_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)
        self.name='jk'
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x, edge_index):
        emb = self.body(x, edge_index)
        x = self.fc(emb)
        return emb,x


class JK_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(JK_Body, self).__init__()
        self.conv1 = GCNConv(nfeat, nhid)
        self.convx = GCNConv(nhid, nhid)
        self.jk = JumpingKnowledge(mode='max')
        self.transition = nn.Sequential(
            nn.ReLU(),
        )

    def forward(self, x, edge_index):
        xs = []
        x = self.conv1(x, edge_index)
        x = self.transition(x)
        xs.append(x)
        for _ in range(1):
            x = self.convx(x, edge_index)
            x = self.transition(x)
            xs.append(x)
        x = self.jk(xs)
        return x