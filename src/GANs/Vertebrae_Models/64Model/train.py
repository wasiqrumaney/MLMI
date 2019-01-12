import torch
from torch import optim
from torch import  nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import transforms, utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py as h5
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, Save_Voxels, generateZ,plotVoxelVisdom
import os
import time
import numpy as np
from data_utils import plot_3d_mesh, VertDataset, ResizeTo
import pathlib

from utils import var_or_cuda, plot_losess
from model import _G, _D
from lr_sh import  MultiStepLR
from visdom import Visdom 
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size = real_samples.size(0)
    lamda = 10 
    alpha = torch.rand(batch_size,1)
    
    #Slicing only a size same as real_samples to avoid conflict with different train batch sizes 
    #fake_samples = fake_samples[:batch_size, :]
    
    #print('alpha {} real {} fake {}'.format(alpha.size(), real_samples.size(), fake_samples.size()))
    if cuda:
        alpha=alpha.cuda()

    interpolates = (alpha * real_samples.data) + ((1 - alpha) * fake_samples.data) #.requires_grad_(True)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    if cuda:
        interpolates=interpolates.cuda()
    
    d_interpolates = D(interpolates)
    d_interpolates = d_interpolates.view(-1,1) ### newly added line
    
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad = True)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True, 
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradient_penalty = (((gradients.norm(2, dim=1) - 1) ** 2)* lamda).mean()
    return gradient_penalty


def train_critic(X, Z, D, G, D_solver, G_solver):
    d_real = D(X)
    fake = G(Z)
    d_fake = D(fake)
    
    # Gradient penalty
    real_batch = Variable(X.type(Tensor) ) 
    gradient_penalty = compute_gradient_penalty(D, real_batch.data, fake.data)
    
    # Adversarial loss
    d_loss = torch.mean(d_fake) - torch.mean(d_real) + gradient_penalty    
    
    Wasserstein_D = torch.mean(d_real - d_fake)
    
    D.zero_grad()
    d_loss.backward()
    D_solver.step()
    G_solver.zero_grad()    
    
    return d_loss, Wasserstein_D, gradient_penalty

def train_gen(Z, D, G, D_solver, G_solver):
    fake = G(Z)
    d_fake = D(fake)
    g_loss = torch.mean(d_fake) * -1
                
    D.zero_grad()
    G.zero_grad()
    g_loss.backward()
    G_solver.step()
    
    return g_loss


def train(args):
    #WSGAN related params
    lambda_gp = 10
    n_critic = 5

    
    hyparam_list = [("model", args.model_name),
                    ("cube", args.cube_len),
                    ("bs", args.batch_size),
                    ("g_lr", args.g_lr),
                    ("d_lr", args.d_lr),
                    ("z", args.z_dis),
                    ("bias", args.bias),
                    ]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(log_param)
    
    #define different paths 
    pickle_path = "." + args.pickle_dir + log_param
    image_path = args.output_dir + args.image_dir + log_param
    pickle_save_path = args.output_dir + args.pickle_dir + log_param


    N=None # None for the whole dataset
    VOL_SIZE = 64
    train_path = pathlib.Path("../Vert_dataset")
    dataset = VertDataset(train_path,n=N , transform=transforms.Compose([ResizeTo(VOL_SIZE),
                                                                   transforms.ToTensor()]))
    print('Number of samples: ',len(dataset))
    dset_loaders = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('Number of batches: ',len(dset_loaders)) 

    
    #  Build the model
    D = _D(args)
    G = _G(args)

    #Create the solvers
    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)


    if torch.cuda.device_count() > 1:
         D = nn.DataParallel(D)
         G = nn.DataParallel(G)
         print("Using {} GPUs".format(torch.cuda.device_count()))
         D.cuda()
         G.cuda()

    elif torch.cuda.is_available():
         print("using cuda")
         D.cuda()
         G.cuda()

    #Load checkpoint if available
    read_pickle(pickle_path, G, G_solver, D, D_solver)

    G_losses=[]
    D_losses=[]

    for epoch in range(args.n_epochs):
        epoch_start_time = time.time() 
        print("epoch %d started" %(epoch))
        for i, X in enumerate(dset_loaders):
            #print(X.shape)
            X = X.view(-1,args.cube_len*args.cube_len*args.cube_len)
            X = var_or_cuda(X)
            X = X.type(torch.cuda.FloatTensor)
            Z = generateZ(num_samples = X.size(0), z_size = args.z_size)
            
            #Train the critic
            d_loss ,Wasserstein_D, gp  = train_critic(X ,Z, D, G, D_solver, G_solver)
            
            # Train the generator every n_critic steps
            if i % n_critic == 0:
                Z = generateZ(num_samples = X.size(0), z_size = args.z_size)
                g_loss = train_gen(Z, D, G, D_solver, G_solver)
            
            #Log each iteration
            iteration = str(G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]]['step'])
            print('Iter-{}; , D_loss : {:.4}, G_loss : {:.4}, WSdistance : {:.4}, GP : {:.4}'.format(iteration, d_loss.item(), \
                                                                            g_loss.item(), Wasserstein_D.item(), gp.item() ))
        ## End of epoch
        epoch_end_time = time.time()
        
        #Plot the losses each epoch
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        plot_losess(G_losses, D_losses, epoch)
        
        if (epoch + 1) % args.image_save_step == 0:
            print("Saving voxels")
            Z = generateZ(num_samples = 8, z_size = args.z_size)
            gen_output = G(Z)
            samples = gen_output.cpu().data[:8].squeeze().numpy()
            samples = samples.reshape(-1,args.cube_len,args.cube_len,args.cube_len)
            Save_Voxels(samples, image_path, iteration)

        if (epoch + 1) % args.pickle_step == 0:
            print("Pickeling the model")
            save_new_pickle(pickle_save_path, iteration, G, G_solver, D, D_solver)

        print("epoch time", (epoch_end_time-epoch_start_time)/60)
        print("epoch %d ended" %(epoch))
        print("################################################")
