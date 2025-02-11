import torch.nn as nn
from neurhci.neurhci_torch import *

"""
    Useful classes above
"""


class Choquet_2_add_Layer(nn.Module):
    """
        creates a layer of {nb} CIs with dimension {dim}
        They all have the same dim-sized input, and return a vector of their
        {nb} outputs
    """
    def __init__(self, dim, nb):
        """
            dim the dim of one aggregator
        """
        super(Choquet_2_add_Layer, self).__init__()
        self.CI = nn.ModuleList(Choquet_2_add(dim) for n in range(nb))
        self.dim = dim
        self.normalize()

    def forward(self, x):
        if len(x.size())==1:
            x = torch.cat(tuple(C(x) for C in self.CI))
        else:
            x = torch.cat(tuple(C(k) for C,k in zip(self.CI,x)))
        return(x)

    def normalize(self):
        for ci in self.CI:
            ci.normalize()

    def normalize_hard(self):
        self.normalize()

    def get_penalization(self):
        pen = torch.tensor([0.]).to(device)
        for i in range(len(self.CI)):
            ci1 = self.CI[i]
            for ci2 in self.CI[i+1:]:
                pen += 1./torch.norm(ci1.output.weight-ci2.output.weight, 2)
        return(pen)

class Matmul_Mod(nn.Module):
    def __init__(self, prev_size, next_size):
        super(Matmul_Mod, self).__init__()
        self.precision = 3.
        self.prev_size = prev_size
        self.next_size = next_size
        self.weight = torch.nn.Parameter(torch.ones(next_size, prev_size).to(device))
        #self.weight.data = torch.rand_like(self.weight)
        """
        while(torch.max(self.weight.data)==0):
            self.weight.data = torch.poisson(torch.ones_like(self.weight))+1
        """
        """
        for i in range(prev_size):
            r_int = torch.randint(0,next_size,(1,))
            self.weight.data[r_int, i] += 1. #making sure we have no zero column
        #self.weight = torch.sigmoid(self.precision*self.pre_weight)
        """
        self.normalize()

    def forward(self, x):
        #x = x.repeat(self.prev_size, 1).T
        #self.weight = torch.sigmoid(self.precision*self.pre_weight)
        return(self.weight*x.to(device))

    def normalize(self):
        with torch.no_grad():
            self.weight.clamp_(0)
            #self.weight = torch.sigmoid(self.precision*self.pre_weight)
            #self.weight.data /= torch.sum(self.weight, dim=0)
            #self.pre_weight.data = (self.weight/(1-self.weight)).log()/self.precision

    def normalize_hard(self):
        with torch.no_grad():
            self.weight.clamp_(0)
            #self.weight = torch.sigmoid(self.precision*self.pre_weight)
            self.weight.data /= torch.sum(self.weight, dim=0)
            #self.pre_weight.data = (self.weight/(1-self.weight)).log()/self.precision

class Fully_Connected_Choquet(nn.Module):
    """
        DAG-type CI network, all CIs from layer l are inputs of everyone in layer n+1
    """
    def __init__(self, dim, shape):
        """
            shape is the list of hidden layer sizes (ex:[3,2])
        """
        super(Fully_Connected_Choquet, self).__init__()
        shape = [dim] + shape + [1]
        self.dim = dim
        layers_list = []
        for lay1, lay2 in zip(shape[:-1], shape[1:]):
            layers_list.append(Choquet_2_add_Layer(lay1, lay2))
        self.layers = nn.Sequential(*layers_list)
        self.normalize()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return(x)

    def normalize(self):
        with torch.no_grad():
            for l in self.layers:
                l.normalize()

class Free_Hierarchy_Choquet(nn.Module):
    """
        Doesn't work
    """
    def __init__(self, dim, shape):
        """
            shape is the list of layer sizes
        """
        super(Free_Hierarchy_Choquet, self).__init__()
        self.inter_layer_weights = []
        self.other_params = []
        shape = [dim] + shape + [1]
        self.dim = dim
        layers_list = []
        for lay1, lay2 in zip(shape[:-1], shape[1:]):
            layers_list.append(Matmul_Mod(lay1, lay2))
            layers_list.append(Choquet_2_add_Layer(lay1, lay2))
            self.inter_layer_weights += list(layers_list[-2].parameters())
            self.other_params += list(layers_list[-1].parameters())
        self.layers = nn.Sequential(*layers_list)
        self.normalize()

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return(x)

    def normalize(self):
        with torch.no_grad():
            for l in self.layers:
                l.normalize()

    def normalize_hard(self):
        with torch.no_grad():
            for l in self.layers:
                l.normalize_hard()