import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DynamicGraphConvolution, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.max = nn.AdaptiveMaxPool1d(1)
        
        self.gc2 = GraphConvolution(136, 30)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*3, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

        self.eps = nn.Parameter(torch.FloatTensor(1))
#         self.eye= nn.Parameter( torch.eye(600))
        self.reset_parameters()
        self.weight = Parameter(torch.Tensor(in_features, 10))



    def reset_parameters(self):
        stdv_eps = 0.1 / math.sqrt(self.eps.size(0))
        nn.init.constant_(self.eps, stdv_eps)

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        
        x_glb = self.gap(x.permute(0,2,1))
        x_max = self.max(x.permute(0,2,1))
        
        a1 = x_glb+x_max
        adj= torch.matmul(a1, a1.permute(0,2,1))

        dynamic_adj = torch.sigmoid(adj)
        
        dynamic_adj =torch.abs(dynamic_adj)
        
#         print(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = F.leaky_relu(x)
        return x

    def forward(self, x):

        dynamic_adj = self.forward_construct_dynamic_graph(x)
#         print("dy",dynamic_adj.shape)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        # print("xxx",x.shape)
        x = self.gc2(x.permute(0,2,1),dynamic_adj)
        return x
