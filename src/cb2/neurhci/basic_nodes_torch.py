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