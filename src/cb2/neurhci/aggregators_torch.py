#%%
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
device = "cpu"


"""
    All structure_dicts are of the form {i:L} where i is the index of a non-leaf node, and L 
    is the list of indices of i's children
    
    The root always has index -1
    
    except for the root, a non-leaf node always has a bigger index than its children
    
    the leaves indices are 0 to n
    
    all indices are positive, consecutive integers
    
    Ex: {-1:[3,4], 3:[0,1], 4:[2,3]} is valid
        {-1:[3,5], 3:[0,1], 5:[2,3]} is not valid 
"""
class Choquet_2_add(nn.Module):
    """
        A single Choquet Integral
    """
    def __init__(self, dim):
        """
            output.weight are the weights that represent the fuzzy measure
            it is a dim*dim vector, such that the first dim terms are the w_i
            then the rest comes as:
            w_1_2_min, w_1_3_min, ..., w_1_n_min, w_2_3_min,...,w_2_n_min,...,w_n-1_n_min,
            w_1_2_max, w_1_3_max, ..., w_1_n_max, w_2_3_max,...,w_2_n_max,...,w_n-1_n_max
        """
        super(Choquet_2_add, self).__init__()
        self.dim = dim
        self.output = nn.Linear(dim*dim, 1, False)
        self.output.weight.data = torch.cat((torch.rand(dim),torch.zeros(dim*(dim-1)))) #initialization as 1-additive
        #self.output.weight.data = torch.cat((torch.poisson(torch.ones(dim))+1,torch.zeros(dim*(dim-1))))
        self.output.weight.data = torch.rand_like(self.output.weight)
        self.output.weight.data = torch.reshape(self.output.weight, [1, dim*dim])
        self.normalize()

    def forward(self, x):
        combs = torch.combinations(x, 2)
        x = torch.cat([x, torch.min(combs, dim=1).values, torch.max(combs, dim=1).values])
        x = self.output(x)
        return(x)

    def normalize(self):
        with torch.no_grad():
            self.output.weight.clamp_(0)
            self.output.weight.data /= torch.sum(self.output.weight)

    def get_mobius(self):
        """
            computes the mobius from the weights, as a vector
        """
        mobius = torch.zeros(self.dim*(self.dim+1)//2).to(device)
        curr_node = self.dim
        with torch.no_grad():
            for i in range(self.dim):
                mobius[i] += self.output.weight[0,i]
                for j in range(i+1, self.dim):
                    wmin = self.output.weight[0,curr_node]
                    wmax = self.output.weight[0,curr_node+int((self.dim*(self.dim-1))/2)]
                    mobius[curr_node] = wmin-wmax
                    mobius[i] += wmax
                    mobius[j] += wmax
                    curr_node += 1
        return(mobius)

    def get_mobius_matrix(self):
        """
            computes the mobius from the weights, as a matrix
        """
        mobius_matrix = torch.zeros((self.dim, self.dim))
        mobius_vector = self.get_mobius()
        curr = self.dim
        for i in range(self.dim):
            mobius_matrix[i,i] = mobius_vector[i]
            for j in range(i+1, self.dim):
                mobius_matrix[j,i] = mobius_vector[curr]
                curr += 1
        return(mobius_matrix+mobius_matrix.T-mobius_matrix.diag()*torch.eye(self.dim))

    def importance_value(self):
        """
            returns the importance values of the children as defined in Labreuche-Fossier IJCAI2018
        """
        mob_mat = self.get_mobius_matrix()
        return(torch.sum(mob_mat, dim=1)/2+(mob_mat.diag()/2))

class Choquet_2_add_ATTENTION(nn.Module):
    """
        A single Choquet Integral
    """
    def __init__(self, dim):
        """
            output.weight are the weights that represent the fuzzy measure
            it is a dim*dim vector, such that the first dim terms are the w_i
            then the rest comes as:
            w_1_2_min, w_1_3_min, ..., w_1_n_min, w_2_3_min,...,w_2_n_min,...,w_n-1_n_min,
            w_1_2_max, w_1_3_max, ..., w_1_n_max, w_2_3_max,...,w_2_n_max,...,w_n-1_n_max
        """
        super(Choquet_2_add_ATTENTION, self).__init__()
        self.dim = dim
        self.pre_weights = nn.Parameter(torch.randn(dim*dim, requires_grad=True))
        self.weights = F.softmax(self.pre_weights, dim=0)

    def forward(self, x):
        if len(x.shape)==1:
            combs = torch.combinations(x, 2)
            x = torch.cat([x, torch.min(combs, dim=1).values, torch.max(combs, dim=1).values])
            self.weights = F.softmax(self.pre_weights, dim=0)
            x = torch.dot(x, self.weights).reshape([1])
        else:
            combs = torch.combinations(torch.arange(0, x.shape[1]), 2)
            xpairs = x[:,combs]
            x_min = torch.min(xpairs, dim=2).values
            x_max = torch.max(xpairs, dim=2).values
            x = torch.cat([x, x_min, x_max], dim=1)
            self.weights = F.softmax(self.pre_weights, dim=0)
            x = F.linear(x, self.weights)
        return(x)
    
    def normalize(self):
        return

    def get_mobius(self):
        """
            computes the mobius from the weights, as a vector
        """
        mobius = torch.zeros(self.dim*(self.dim+1)//2).to(device)
        curr_node = self.dim
        with torch.no_grad():
            for i in range(self.dim):
                mobius[i] += self.weights[0,i]
                for j in range(i+1, self.dim):
                    wmin = self.weights[0,curr_node]
                    wmax = self.weights[0,curr_node+int((self.dim*(self.dim-1))/2)]
                    mobius[curr_node] = wmin-wmax
                    mobius[i] += wmax
                    mobius[j] += wmax
                    curr_node += 1
        return(mobius)

    def get_mobius_matrix(self):
        """
            computes the mobius from the weights, as a matrix
        """
        mobius_matrix = torch.zeros((self.dim, self.dim))
        mobius_vector = self.get_mobius()
        curr = self.dim
        for i in range(self.dim):
            mobius_matrix[i,i] = mobius_vector[i]
            for j in range(i+1, self.dim):
                mobius_matrix[j,i] = mobius_vector[curr]
                curr += 1
        return(mobius_matrix+mobius_matrix.T-mobius_matrix.diag()*torch.eye(self.dim))

    def importance_value(self):
        """
            returns the importance values of the children as defined in Labreuche-Fossier IJCAI2018
        """
        mob_mat = self.get_mobius_matrix()
        return(torch.sum(mob_mat, dim=1)/2+(mob_mat.diag()/2))

# %%
