import torch
import torch.nn as nn
from cb2.neurhci.aggregators_torch import *
from cb2.neurhci.marginal_utilities_torch import *

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
def get_dim(hierar):
    """
        returns the nb of leaves of the structure
    """
    if len(hierar)>1:
        return(min(k for k in hierar if k>0))
    else:
        return(len(hierar[-1]))

def get_height(hierarchy, node=-1, heights = {}):
    """
        given a tree, returns the pairs {node: height}
        height is the longest distance from a leaf
    """
    if (node==-1):
        heights = {}
    if node not in hierarchy:
        heights[node] = 0
    else:
        for c in hierarchy[node]:
            get_height(hierarchy, c, heights)
        heights[node] = max(heights[c] for c in hierarchy[node])+1
    return(heights)

class HCI_2_add(nn.Module):
    """
        An HCI without marginal utilities
    """
    def __init__(self, hierarchy):
        super(HCI_2_add, self).__init__()
        self.dim = get_dim(hierarchy)
        self.hierarchy = hierarchy
        self.inputs = {h:torch.zeros(len(c)) for (h,c) in hierarchy.items()}
        self.CIs = nn.ModuleDict({str(h):Choquet_2_add_ATTENTION(len(c)) for (h,c) in hierarchy.items()})
        self.height_map = {}
        heights = get_height(hierarchy)
        for (k,v) in heights.items():
            self.height_map.setdefault(v, []).append(k)
        self.height = max(self.height_map)
        self.normalize()

    def forward(self, x):
        self.inputs = torch.cat([x, torch.zeros(len(self.hierarchy)).to(device)])
        for h in range(1, self.height+1):
            agregs = self.height_map[h]
            for ag in agregs:
                self.inputs[ag] = self.CIs[str(ag)](torch.cat([self.inputs[c].reshape((1,1)) for c in self.hierarchy[ag]], dim=1).to(device)[0])
        x = self.inputs[-1].reshape((1,))
        return(x)

    def normalize(self):
        with torch.no_grad():
            for CI in self.CIs.values():
                CI.normalize()

    def get_path(self, node):
        """
            get the list of ancestors of node ordered from the root
        """
        assert(node in list(range(self.dim))+list(self.hierarchy.keys()))
        path = [node]
        curr_node = node
        while not (-1 in path):
            for (key, val) in self.hierarchy.items():
                if curr_node in val:
                    curr_node = key
                    break
            path.append(curr_node)
        return(path[::-1])

    def get_all_paths(self, curr_node=-1, paths={}):
        """
            TODO
        """
        print(3)

    def get_ieow_leaf_global(self, node):
        curr_ieow = 1.
        path = self.get_path(node)
        for i in range(len(path)-1):
            curr_node = self.CIs[str(path[i])]
            curr_ieow *= curr_node.importance_value()[self.hierarchy[path[i]].index(path[i+1])]
        return(curr_ieow)

    def get_2_add_shapleys(self):
        """
            #TODO : recursify
        """
        return([self.get_ieow_leaf_global(i) for i in range(self.dim)])

class HCI_2_add_2_layers_partition(HCI_2_add):
    """
        An HCI without marginal utilities
    """
    def __init__(self, whites_by_aggregator, dim_white):
        nb_intermediate_concepts = dim_white // whites_by_aggregator
        if dim_white % whites_by_aggregator:
            nb_intermediate_concepts += 1
        indices_intermediate = list(range(dim_white, dim_white+nb_intermediate_concepts))
        hierarchy = {inter_index:list(range(i*whites_by_aggregator, min((i+1)*whites_by_aggregator, dim_white)))
                                    for i, inter_index in enumerate(indices_intermediate)}
        hierarchy[-1] = indices_intermediate
 
        HCI_2_add.__init__(self, hierarchy)
        self.lower_level = [self.CIs[str(i)] for i in hierarchy if i!=-1]
        self.inputs_top_down = [(min(hierarchy[k]), max(hierarchy[k])) for k in self.hierarchy if k !=-1]
        self.agg_top = self.CIs["-1"]

    def forward(self, x):
        if len(x.shape) == 1:
            x = torch.cat([C(x[z[0]:z[1]+1]) for C,z in zip(self.lower_level, self.inputs_top_down)])
            x = self.agg_top(x)
        else:
            m = torch.cat([C(x[:,z[0]:z[1]+1]).unsqueeze(-1) for C,z in zip(self.lower_level, self.inputs_top_down)], dim=1)
            x = self.agg_top(m)
        return({'output': x.squeeze(-1), 'snz':m})

    def normalize(self):
        return

class HCI_2_add_balanced(HCI_2_add):
    def __init__(self, m, n):
        hierarchy = {}
        curr_lay = list(range(n))
        all_lays = []
        while len(curr_lay)>m:
            lcl = len(curr_lay)
            good = lcl//m
            rest = lcl%m
            new_lay = list(range(max(curr_lay)+1, max(curr_lay)+1 + good + int(rest!=0)))
            for i,l in enumerate(new_lay):
                hierarchy[l] = curr_lay[i*m: min(i*m+m, len(curr_lay))]
            curr_lay = new_lay
            all_lays.append(curr_lay)
        hierarchy[-1] = curr_lay
        all_lays.append([-1])

        HCI_2_add.__init__(self, hierarchy)

        self.leaves = n
        self.children_per_node = m
        self.layers = all_lays
        self.layers_CIs = [[self.CIs[str(k)] for k in lay] for lay in self.layers]
        self.inputs_top_down = [(min(hierarchy[k]), max(hierarchy[k])) for k in self.hierarchy]

    def forward(self, x):
        if len(x.shape) == 1:
            for lay in self.layers_CIs:
                x = torch.cat([C(x[i*self.children_per_node: min((i+1)*self.children_per_node, len(x))]) for i,C in enumerate(lay)])
        else:
            for i,lay in enumerate(self.layers_CIs):
                x = torch.cat([C(x[:,i*self.children_per_node: min((i+1)*self.children_per_node, x.shape[1])]).unsqueeze(-1) for i,C in enumerate(lay)], dim=1)
                if not i:
                    m=x
        return({'output':x.squeeze(-1), 'snz':m})

class NeurHCI(nn.Module):
    """
        An HCI with given hierarchy, and increasing utilities
        
        TODO : select the vector of monotonicity
    """
    def __init__(self, hierarchy, leaves_types={}, nb_prim=200):
        """
            leaf_nodes has the shape {i: type of leaf} for i in range(dim)
            Leaves types are:
        """
        types_dict = {"UI": Utility_Increasing, "UIA": Utility_Increasing_ATTENTION, "UD": Utility_Decreasing, "UDA": Utility_Decreasing_ATTENTION, "ID": Identity_Leaf, "Nneg": One_Minus_Identity_Leaf, "M": Monot_Selector, "MA": Monot_Selector_ATTENTION, "IN": ID_NEG_Selector}
        super(NeurHCI, self).__init__()
        self.dim = get_dim(hierarchy)
        for i in range(self.dim):
            leaves_types.setdefault(i, "ID")
        self.utils_list = nn.ModuleList([types_dict[leaves_types[d]](nb_prim) for d in range(self.dim)])
        self.agreg = HCI_2_add(hierarchy)

    def utils_propag(self, x):
        utils = [ui(xi).unsqueeze(-1) for ui,xi in zip(self.utils_list, x)]
        return(torch.cat(utils))

    def get_leaves_shapley(self):
        return(self.agreg.get_2_add_shapleys())

    def forward(self, x):
        if len(x.shape)==1:
            x = self.utils_propag(x)
            x = self.agreg(x)
        else:
            x = torch.cat([self.agreg(self.utils_propag(u)) for u in x])
        return(x)

    def normalize(self):
        self.agreg.normalize()
        for ut in self.utils_list:
            ut.normalize()

class NeurHCI_2_layers(NeurHCI):
    """
        An HCI with given hierarchy, and increasing utilities
        
        TODO : select the vector of monotonicity
    """
    def __init__(self, whites_by_aggregator, dim_white, leaves_types={}, nb_prim=200):
        """
            leaf_nodes has the shape {i: type of leaf} for i in range(dim)
            Leaves types are:
        """
        agreg = HCI_2_add_2_layers_partition(whites_by_aggregator, dim_white)
        super(NeurHCI_2_layers, self).__init__(agreg.hierarchy, leaves_types, nb_prim)
        self.agreg = agreg

class NeurHCI_Balanced(NeurHCI):
    """
        An HCI with given hierarchy, and increasing utilities
        
        TODO : select the vector of monotonicity
    """
    def __init__(self, whites_by_aggregator, dim_white, leaves_types={}, nb_prim=200):
        """
            leaf_nodes has the shape {i: type of leaf} for i in range(dim)
            Leaves types are:
        """
        agreg = HCI_2_add_balanced(whites_by_aggregator, dim_white)
        super(NeurHCI_Balanced, self).__init__(agreg.hierarchy, leaves_types, nb_prim)
        self.agreg = agreg
