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
class Leaf_Node(nn.Module):
    def __init__(self, nb_prim=0.):
        super(Leaf_Node, self).__init__()
        self.nb_prim = nb_prim
    
    def normalize(self):
        return
        
    def plot(self):
        X = torch.linspace(0,1,1000).reshape((1000,1)).to(device)
        y = torch.tensor([self(x).to('cpu') for x in X])
        return(X.reshape((1000,)),y.reshape((1000,)))

class Identity_Leaf(Leaf_Node):
    """
        identity marginal utility
    """
    def __init__(self, nb_prim=0.):
        super(Identity_Leaf, self).__init__()

    def forward(self, x):
        return(x)

class One_Minus_Identity_Leaf(Leaf_Node):
    """
        1-x marginal utility
    """
    def __init__(self, nb_prim=0.):
        super(One_Minus_Identity_Leaf, self).__init__()

    def forward(self, x):
        return(1-x)
    
class Selector(Leaf_Node):
    def __init__(self, t1, t2, nb_prim=100):
        super(Selector, self).__init__()
        self.first = t1
        self.secon = t2
        self.chooser = nn.Parameter(torch.tensor(0.))
        self.switch = torch.sigmoid(self.chooser)

    def forward(self, x):
        self.switch = torch.sigmoid(self.chooser)
        return(self.switch*self.first(x) + (1-self.switch)*self.secon(x))
    
    def normalize(self):
        print("oui oui")
        self.first.normalize()
        self.secon.normalize()

class ID_NEG_Selector(Selector):
    def __init__(self, nb_prim=0):
        first = Identity_Leaf()
        second = One_Minus_Identity_Leaf()
        Selector.__init__(self, first, second)

class Monot_Selector(Selector):
    def __init__(self, nb_prim=0):
        first = Utility_Increasing(nb_prim)
        second = Utility_Decreasing(nb_prim)
        Selector.__init__(self, first, second)

    def forward(self, x):
        self.switch = torch.sigmoid(10.*self.chooser)
        return(self.switch*self.first(x) + (1-self.switch)*(1-self.first(x)))

class Utility_Increasing(Leaf_Node):
    """
        A module representing a non-decreasing convex sum of sigmoids
    """
    def __init__(self, nb_prim=100):
        """
            nb_prim : nb of sigmoids
            bias_layer : the bias of each sigmoid
            etas_layer : the precision of each sigmoid
        """
        super(Utility_Increasing, self).__init__()

        self.bias_layer = nn.Linear(1, nb_prim, True)
        self.etas_layer = nn.Linear(nb_prim, nb_prim, False)
        self.output = nn.Linear(nb_prim, 1, False)

        self.bias_layer.weight.data = torch.ones(self.bias_layer.weight.shape)
        self.bias_layer.weight.requires_grad_(False)
        self.bias_layer.bias.data = -torch.rand(self.bias_layer.bias.shape).to(device)

        self.etas_layer.weight.data = (torch.rand(self.etas_layer.weight.shape)*nb_prim).to(device)

        self.output.weight.data = torch.ones(self.output.weight.shape).to(device)
        self.normalize()

    def forward(self, x):
        x = self.bias_layer(x)
        x = torch.sigmoid(self.etas_layer(x))
        x = self.output(x)
        return(x)

    def normalize(self):
        """
            ensure the constraints
            must be called after each optimize.step()
        """
        print("oui")
        with torch.no_grad():
            self.etas_layer.weight.clamp_(0)
            self.etas_layer.weight.data *= torch.eye(self.nb_prim).to(device)
            self.output.weight.data = torch.where(
                                        self.bias_layer.bias>-0,
                                        torch.zeros(1).to(device)[0],
                                        self.output.weight.data).to(device)
            self.output.weight.data = torch.where(
                                        self.bias_layer.bias<-1,
                                        -torch.ones(1).to(device)[0],
                                        self.output.weight.data).to(device)
            self.output.weight.clamp_(0)
            self.output.weight /= torch.sum(self.output.weight)
            self.bias_layer.bias.clamp_(-1,0)

class Utility_Decreasing(Utility_Increasing):
    """
        Non-increasing marginal utility
    """
    def __init__(self, nb_prim):
        Utility_Increasing.__init__(self, nb_prim)

    def forward(self, x):
        x = Utility_Increasing.forward(self, x)
        return(1-x)

class Utility_Increasing_ATTENTION(Leaf_Node):
    """
        A module representing a non-decreasing convex sum of sigmoids
    """
    def __init__(self, nb_prim=100):
        """
            nb_prim : nb of sigmoids
            bias_layer : the bias of each sigmoid
            etas_layer : the precision of each sigmoid
        """
        Leaf_Node.__init__(self, nb_prim)
        self.nb_prim = nb_prim 
        self.pre_precision = nn.Parameter(torch.rand(nb_prim))*self.nb_prim
        self.bias          = (torch.linspace(1/nb_prim, 1-1/nb_prim, nb_prim))
        self.pre_weights   = nn.Parameter(torch.randn(nb_prim)*0.1+torch.log(torch.tensor(2.)))
        self.weights = F.softmax(self.pre_weights, dim=0)
        self.precision = F.relu(self.pre_precision)

    def forward(self, x):
        self.precision = F.relu(self.pre_precision)
        self.weights = F.softmax(self.pre_weights, dim=0)
        x = x-self.bias
        x = x*self.precision
        x = torch.sigmoid(x)
        x = torch.dot(x, self.weights)
        return(x)

class Utility_Decreasing_ATTENTION(Utility_Increasing_ATTENTION):
    """
        Non-increasing marginal utility
    """
    def __init__(self, nb_prim):
        Utility_Increasing_ATTENTION.__init__(self, nb_prim)

    def forward(self, x):
        x = Utility_Increasing_ATTENTION.forward(self, x)
        return(1-x)

class Monot_Selector_ATTENTION(Selector):
    def __init__(self, nb_prim=0):
        self.steps = 0.
        self.direction = 0
        first = Utility_Increasing_ATTENTION(nb_prim)
        second = Utility_Decreasing_ATTENTION(nb_prim)
        Selector.__init__(self, first, second)

    def normalize(self):
        direction_now = int(self.chooser>0)
        if self.direction == direction_now:
            self.steps += 1
        else:
            self.direction = 1-self.direction
            self.steps = 0
        if self.steps>10:
            ...
        return