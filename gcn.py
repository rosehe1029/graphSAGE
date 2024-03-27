import torch 
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def  __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):    #x特征矩阵,agj邻接矩阵 
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
        
        
        
        
        
from torch_geometric.nn import GCNConv,SAGEConv,GATConv 

class GCN(torch.nn.Module):
    def __init__(self,f,h,c):
        super().__init__()
        self.conv1=GCNConv(f,h)
        self.conv2=GCNConv(h,c)
    def forward(self,data):
        x,edge_index=data.x,data.edge_index      #edge_index为PYG邻接矩阵格式的数据
        x=self.conv1(x,edge_index)
        x=F.relu(x)
        x=F.dropout(x,training=self.training)
        x=self.conv2(x,edge_index)
        
        return x
    
    
    
    
    