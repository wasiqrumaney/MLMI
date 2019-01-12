import scipy.ndimage as nd
import scipy.io as io
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle
import numpy as np 

def plot_losess(G_losses, D_losses, epoch_number):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.plot(np.arange(len(G_losses)),G_losses,label='G_loss')
    plt.plot(np.arange(len(D_losses)),D_losses,label='Critic_loss')
    plt.legend(loc='upper left')
    plt.xlabel("Number of epochs")
    plt.ylabel("Losses")
    plt.savefig("plots/losses_{}.png".format(epoch_number))
    plt.close()



def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def plotFromVoxels(voxels):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.show()


def Save_Voxels(voxels, path, iteration):
    if not os.path.exists(path):
        os.makedirs(path)
    
    #np.save("voxels_before.npy",voxels)
    voxels = voxels[:8].__ge__(0.5)
    #np.save("voxels_after.npy",voxels)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()

    with open(path + '/{}.pkl'.format(str(iteration).zfill(3)), "wb") as f:
        pickle.dump(voxels, f, protocol=pickle.HIGHEST_PROTOCOL)


def make_hyparam_string(hyparam_dict):
    str_result = ""
    for i in hyparam_dict.keys():
        str_result = str_result + str(i) + "=" + str(hyparam_dict[i]) + "_"
    return str_result[:-1]


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def generateZ( num_samples, z_size=100, dist='norm'):

    if dist == "norm":
        Z = var_or_cuda(torch.Tensor(num_samples, z_size).normal_(0, 1))
    elif dist == "uni":
        Z = var_or_cuda(torch.randn(num_samples, z_size))
    else:
        print("z_dist is not normal or uniform")

    return Z

########################## Pickle helper ###############################


def read_pickle(path, G, G_solver, D_, D_solver):
    try:

        files = os.listdir(path)
        file_list = [int(file.split('_')[-1].split('.')[0]) for file in files]
        file_list.sort()
        recent_iter = str(file_list[-1])
        print(recent_iter, path)

        with open(path + "/G_" + recent_iter + ".pkl", "rb") as f:
            G.load_state_dict(torch.load(f))
            print("Done loading G {:.4}".format(recent_iter))
        with open(path + "/G_optim_" + recent_iter + ".pkl", "rb") as f:
            print("Done loading G_Opti {:.4}".format(recent_iter))
            G_solver.load_state_dict(torch.load(f))
        with open(path + "/D_" + recent_iter + ".pkl", "rb") as f:
            print("Done loading D {:.4}".format(recent_iter))
            D_.load_state_dict(torch.load(f))
        with open(path + "/D_optim_" + recent_iter + ".pkl", "rb") as f:
            print("Done loading D_Optim {:.4}".format(recent_iter))
            D_solver.load_state_dict(torch.load(f))


    except Exception as e:

        print("fail try read_pickle", e)



def save_new_pickle(path, iteration, G, G_solver, D_, D_solver):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + "/G_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G.state_dict(), f)
    with open(path + "/G_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(G_solver.state_dict(), f)
    with open(path + "/D_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_.state_dict(), f)
    with open(path + "/D_optim_" + str(iteration) + ".pkl", "wb") as f:
        torch.save(D_solver.state_dict(), f)
