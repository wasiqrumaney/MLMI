# import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.io as pio
from skimage.measure import marching_cubes_lewiner
from skimage.transform import resize
import numpy as np
from math import floor, ceil
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from torch.nn import init
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D as ax3
import psutil

# init_notebook_mode(connected=True)
init_notebook_mode()


############################# VISUALIZATION UTILS ######################
def plot_3d_sum(voxels, axes=[0, 1, 2]):
    if isinstance(voxels, torch.Tensor):
        if voxels.is_cuda:
            voxels = voxels.cpu()
        voxels = voxels.numpy()
    vol = voxels.squeeze()

    plt.figure()
    n = len(axes)
    plt.subplots_adjust(wspace=0, hspace=0)
    for i in range(n):
        img2d = np.sum(vol, axis=axes[i])
        plt.subplot(1, n, i + 1)
        plt.axis("off")
        plt.imshow(img2d)

def save_3d_mesh_single_view(voxels, color='#26638F', opacity=0.5, views=[[1.25,1.25,1.25]] ):
    """
    Plots the 3d volume as mesh using Plotly.
    :param voxels: volume as voxels. can be Tensor or np.ndarray
    :param color: rba code of the color
    :param opacity: opacity value
    :param views: list of list with x, y, z, eg. views=[[1.25,1.25,1.25],[1,1,0.1]]
    """
    if isinstance(voxels, torch.Tensor):
        if voxels.is_cuda:
            voxels = voxels.cpu()
        voxels = voxels.numpy()
    voxels = voxels.squeeze()
    # do padding in order to avoid surface on the sides not showing
    v, f, _, _ = marching_cubes_lewiner(np.pad(voxels, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant'))

    x = v[:, 0].tolist()
    y = v[:, 1].tolist()
    z = v[:, 2].tolist()
    i = f[:, 0].tolist()
    j = f[:, 1].tolist()
    k = f[:, 2].tolist()
    data = [go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity)]
    layout = go.Layout(
        # scene=scene1,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)

    if not os.path.exists('./images'):
        os.mkdir('images')
    for i in range(len(views)):
        view = views[i]
        eye = dict(
            x = view[0],
            y = view[1],
            z = view[2]
        )
        camera = dict(
            # up=dict(x=0, y=0, z=1),
            # center=dict(x=2, y=2, z=0.1),
            eye=eye
        )
        fig['layout'].update(
            scene=dict(camera=camera),
        )
        pio.write_image(fig, './images/mesh{}.png'.format(i))



def plot_3d_mesh(voxels, color='#26638F', opacity=0.5):
    """
    Plots the 3d volume as mesh using Plotly.
    :param voxels: volume as voxels. can be Tensor or np.ndarray
    :param color: rba code of the color
    :param opacity: opacity value

    """
    if isinstance(voxels, torch.Tensor):
        if voxels.is_cuda:
            voxels = voxels.cpu()
        voxels = voxels.numpy()
    voxels = voxels.squeeze()
    # do padding in order to avoid surface on the sides not showing
    v, f, _, _ = marching_cubes_lewiner(np.pad(voxels, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant'))

    x = v[:, 0].tolist()
    y = v[:, 1].tolist()
    z = v[:, 2].tolist()
    i = f[:, 0].tolist()
    j = f[:, 1].tolist()
    k = f[:, 2].tolist()

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=2, y=2, z=0.1)
    )
    scene1 = dict(camera=camera)
    trace = go.Mesh3d( x=x, y=y, z=z, i=i, j=j, k=k, color=color, opacity=opacity )
    iplot([trace])


def plot_voxels_single_view(vol, elev_azim=(30, 45)):
    fig = plt.figure(figsize=(10, 10))
    facecolors = cm.Blues(vol)
    facecolors[:, :, :, -1] = 0.5  # vol

    ax = fig.gca(projection='3d')
    ax.voxels(vol, facecolors=facecolors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(*elev_azim)
    ## Colorbar
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbaxes = fig.add_axes([0.9, 0.3, 0.03, 0.5])
    cb1 = mpl.colorbar.ColorbarBase(ax=cbaxes, cmap=cmap, norm=norm, orientation='vertical')
    plt.show()


def plot_voxels_views(vol, views=[(30, 0), (30, 45), (30, 90), (30, 135), (30, 270), (30, 315)]):
    N_views = len(views)
    for i, elev_azim in enumerate(views):
        plot_voxels_single_view(vol, elev_azim)


def compare_voxels_single_view(vol1, vol2, elev_azim=(30,45)):
    fig = plt.figure(figsize=(20, 10))
    ##### VOLUME 1 ######
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    colors = np.zeros(vol2.shape + (4,))
    # colors[:, :, :, 2] = 1
    colors = cm.Blues(vol1)
    colors[:, :, :, -1] = 0.5
    ax.voxels(vol1, facecolors=colors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(*elev_azim)
    ##### VOLUME 2 #######
    facecolors = cm.Blues(vol2)
    facecolors[:, :, :, -1] = 0.5  # vol2
    # facecolors = np.zeros(vol2.shape + (4,)) + 0.5
    # facecolors[:,:,:,2] = 1-vol2[:,:,:]
    # # facecolors[:,:,:,3] = vol2[:,:,:]

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.voxels(vol2, facecolors=facecolors)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(*elev_azim)
    ## Colorbar
    cmap = mpl.cm.Blues
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cbaxes = fig.add_axes([0.5, 0.3, 0.03, 0.5])
    cb1 = mpl.colorbar.ColorbarBase(ax=cbaxes, cmap=cmap, norm=norm, orientation='vertical')
    plt.show()


def compare_voxels_views(vol1, vol2, views=[(40, 0), (40, 45), (40, 90), (40, 135), (40, 270), (40, 315)]):
    N_views = len(views)
    for i, elev_azim in enumerate(views):
        compare_voxels_single_view(vol1, vol2, elev_azim)


############################# MODEL / DATA UTILS ######################
def init_weights(m):
    if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def return_init_weights(init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    return init_func


class VertDataset(Dataset):
    '''
    Dataset for the vertebrae dataset as npz files. With param 'n' uses only the first n files.
    '''

    def __init__(self, root_dir, n=None, transform=None, region=None):
        assert os.path.isdir(root_dir)
        self.root_dir = root_dir
        self.files_list = [filename for filename in os.listdir(root_dir)]
        if region:
            if region is 'thoracic':
                self.idx_min = 8
                self.idx_max = 19
            elif region is 'lumbar':
                self.idx_min = 20
                self.idx_max = 24
            else:
                raise Exception("Wrong value for parameter 'region'")
            self.files_list = [file for file in self.files_list if
                               self.idx_min <= int(file.split(".")[0].split("_")[-1]) <= self.idx_max]
        # print(self.files_list)
        if n:
            self.files_list = self.files_list[:n]

        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files_list[idx])
        sample = np.load(file_path)['vert_msk_true_res']

        if self.transform:
            sample = self.transform(sample)

        return sample.unsqueeze(0).float()


class ResizeTo(object):
    '''
    Transform that resizes to size specified at initialization
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, mask):
        max_dim = max(max(mask.shape), self.output_size)
        pad_by_dim = [(int(floor((max_dim - k) / 2)), int(ceil((max_dim - k) / 2))) for k in mask.shape]
        mask = np.pad(mask, pad_width=pad_by_dim, mode='constant', constant_values=0)
        if max_dim != self.output_size:
            new_size = (self.output_size, self.output_size, self.output_size)
            mask = resize(mask, new_size, order=0, preserve_range=True,
                          clip=True, anti_aliasing=False, mode="constant")

        return mask


def do_threshold(nda, eps=0.5, do_ceil=False, out_type='numpy'):
    if out_type is 'tensor':
        pass
    else:
        if isinstance(nda, torch.Tensor):
            if nda.is_cuda:
                nda = nda.cpu()
            nda = nda.numpy()
        nda = nda.squeeze()
        nda[nda < eps] = 0
        if do_ceil:
            nda[np.nonzero(nda)] = 1
        nda.squeeze()
    return nda


############################# TRAINING UTILS #####################
def save_checkpoint(results_dir, epoch, model, optimizer, train_logger, rec_logger, kl_logger, val_logger):
    filename = 'cp_e%d' % (epoch + 1)
    checkpoint_path = '%s/%s' % (results_dir, filename)

    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'train_logs': train_logger,
             'rec_logs': rec_logger, 'kl_logs': kl_logger, 'val_logs': val_logger}
    torch.save(state, checkpoint_path)


def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        train_logger = checkpoint['train_logs']
        rec_logger = checkpoint['rec_logs']
        kl_logger = checkpoint['kl_logs']
        val_logger = checkpoint['val_logs']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, train_logger, rec_logger, kl_logger, val_logger

def load_checkpoint_old(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        train_logger = checkpoint['train_logs']
        rec_logger = checkpoint['rec_logs']
        kl_logger = checkpoint['kl_logs']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        raise Exception("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, train_logger, rec_logger, kl_logger
