import torch
import torch.nn as nn

class EdgeEncoder(nn.Module):
    def __init__(self, in_edge_dim, d):
        super().__init__()
        self.d = d
        self.W = nn.Linear(in_edge_dim,d*d)

    def forward(self, edge_attr):
        A = self.W(edge_attr)
        A = A.view(-1,self.d,self.d)
        A = A - torch.transpose(A, -1, -2)
        A = torch.linalg.matrix_exp(A)
        return A