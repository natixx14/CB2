#%%
import neurhci.neurhci_torch as neurhci
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

#D = {-1:[4,5], 4:[0,1], 5:[2,3]}
D = {-1:[4,5], 4:[0,1,2,3], 5:[0,1,2,3]}
TD = {0:"MA", 1:"MA", 2:"MA", 3:"MA"}

N = neurhci.NeurHCI(D, TD, nb_prim=100)
N = N.float()

X = [np.random.rand(32, 4) for _ in range(10)]

def u0(x):
    return((x)**5)

def u1(x):
    return(x**2)

def u2(x):
    return(1-np.sqrt(x))

def u3(x):
    if x<0.25:
        return(1-3*x)
    else:
        return(0.25-(x-0.25)/3)

u = [u0, u1, u2, u3]
#u = [lambda x: x for _ in range(2)] + [lambda x: 1-x for _ in range(2)]

def CI(x):
    #return(max(min(x[0], x[1]), min(1-x[2], 1-x[3])))
    return(0.15*u[0](x[0]) + 0.2*u[1](x[1]) + 0.05*min(u[0](x[0]), u[1](x[1])) + 0.35*min(u[0](x[0]), u[2](x[2])) + 0.25*min(u[2](x[2]), u[3](x[3])))

Y = [[CI(x) for x in batch] for batch in X]
inputs = torch.tensor(X)
targets = torch.tensor(Y)

def plot_utils(N):
    for i in range(4):
        plt.subplot(2,2,i+1)
        #outs = ([N.utils_list[i](x).   h() for x in Xs])
        X,Y = N.utils_list[i].plot()
        x0 = 0
        plt.plot(np.linspace(x0,x0+1,100), [u[i](x) for x in np.linspace(x0,x0+1,100)], marker='o')
        plt.plot(X, Y)
        #plt.plot(Xs.detach(), [o[0] for o in outs])

def train_vanilla(net, EPOCHS_TO_TRAIN=50, batch_size=1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
            {'params': net.utils_list.parameters()},
            {'params': net.agreg.parameters(), 'lr': 0.01}
        ], lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    data_train = list(zip(inputs, targets))
    print("Training loop:")
    for idx in range(0, EPOCHS_TO_TRAIN):
        #if idx in [0,1,2,10,20,50,100,150,200,250]:
        #    plot_utils(net)
        #for batch in batches:
        #for input, target in batch:
        for input, target in data_train:
            #loss = torch.zeros(1).reshape([])
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(input.float()).reshape([-1])
            loss = criterion(output, target.float())
            loss.backward(retain_graph=True)
            optimizer.step()    # Does the update
        plot_utils(N)
        #plt.legend()
        plt.show()
        scheduler.step()
        net.normalize()
        if idx % 1 == 0:
            print("Epoch {: >8} Loss: {}".format(idx, loss.data))
    net.normalize()

train_vanilla(N, 200)

GT = [0.15, 0.2, 0., 0., 0.05, 0.35, 0., 0., 0., 0.25]
Z = N.agreg.CIs['-1'].get_mobius()
for z,gt in zip(Z,GT): 
    print("----------------------------")
    print("Obtained:")
    print("{:0.2f}".format(z))
    print("Expected:")
    print("{:0.2f}".format(gt))

    
plot_utils(N)
plt.legend()
plt.show()

# %%