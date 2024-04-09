import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F       
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils import data
import os
import math
import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import torch.utils.data as data


def gen_pendulum(data_dir = 'mini_pendulum',noise_type='normal',noise_level=0.1,grayscale=True):

    def projection(theta, phi, x, y, base = -0.5):
        b = y-x*math.tan(phi)
        shade = (base - b)/math.tan(phi)
        return shade

    if not os.path.exists(f'./causal_data/{data_dir}/'): 
        os.makedirs(f'./causal_data/{data_dir}/train/')
        os.makedirs(f'./causal_data/{data_dir}/test/')

    count = 0
    df_labels = []

    for i in tqdm(range(-44,44)):#pendulum
        for j in range(60,140):#light
            if j == 100: continue

            # Get phi, theta and then ball position
            plt.rcParams['figure.figsize'] = (1.0, 1.0)
            theta = i*math.pi/200.0
            phi = j*math.pi/200.0
            x = 10 + 8*math.sin(theta)
            y = 10.5 - 8*math.cos(theta)

            ball = plt.Circle((x,y), 1.5, color = 'firebrick')
            gun = plt.Polygon(([10,10.5],[x,y]), color = 'black', linewidth = 3)

            # Sun Position and size
            light = projection(theta, phi, 10, 10.5, 20.5)
            sun = plt.Circle((light,20.5), 3, color = 'orange')

            #calculate the mid index of shade
            ball_x = 10+9.5*math.sin(theta)
            ball_y = 10.5-9.5*math. cos(theta)
            mid = (projection(theta, phi, 10.0, 10.5)+projection(theta, phi, ball_x, ball_y))/2
            shade = max(3,abs(projection(theta, phi, 10.0, 10.5)-projection(theta, phi, ball_x, ball_y)))

            # Additive noise for endogenous vars (if any)
            if noise_type == 'normal':
                shade = np.random.normal(shade, noise_level)
                mid = np.random.normal(mid, noise_level)
            elif noise_type == 'uniform':
                shade = np.random.uniform(shade-noise_level, shade+noise_level)
                mid = np.random.uniform(mid-noise_level, mid+noise_level)

            shadow = plt.Polygon(([mid - shade/2.0, -0.5],[mid + shade/2.0, -0.5]), color = 'black', linewidth = 3)
            
            ax = plt.gca()
            ax.add_artist(gun)
            ax.add_artist(ball)
            ax.add_artist(sun)
            ax.add_artist(shadow)
            ax.set_xlim((0, 20))
            ax.set_ylim((-1, 21))
            plt.axis('off')

            df_labels.append([i,j,shade,mid])

            if count == 4: 
                sample_type = 'test'
                count = 0
            else: 
                sample_type = 'train'

            count += 1

            # Save file as png, convert to grayscale
            save_path  = './causal_data/{}/{}/a_{}_{}_{}_{}.png'.format(data_dir, sample_type, int(i), int(j), int(shade), int(mid))
            plt.savefig(save_path,dpi=50)
            if grayscale:
                img = Image.open(save_path)
                img.convert("L").save(save_path)
                
            plt.clf()

    df_labels = pd.DataFrame(df_labels,columns=['i', 'j', 'shade','mid'])
    df_labels.to_csv(f'./causal_data/{data_dir}/labels.csv',index=False)
    print('Data generated in ./causal_data/{}'.format(data_dir))

    return df_labels
 
def gen_pendulum_tabular(file_name,noise_type='normal',noise_level=0.1,shade_cut = False):

    if not os.path.exists('./causal_data/'): 
        os.makedirs('./causal_data/')

    count = 0
    df_labels = []

    for i in range(-44,44):#pendulum
        for j in range(60,140):#light
            if j == 100: continue

            # Get phi, theta and then ball position
            plt.rcParams['figure.figsize'] = (1.0, 1.0)
            theta = i*math.pi/200.0
            phi = j*math.pi/200.0
            x = 10 + 8*math.sin(theta)
            y = 10.5 - 8*math.cos(theta)

            #calculate the mid index of shade
            ball_x = 10+9.5*math.sin(theta)
            ball_y = 10.5-9.5*math. cos(theta)
            mid = (projection(theta, phi, 10.0, 10.5)+projection(theta, phi, ball_x, ball_y))/2
            if shade_cut:
                shade = max(3,abs(projection(theta, phi, 10.0, 10.5)-projection(theta, phi, ball_x, ball_y)))
            else:
                shade = abs(projection(theta, phi, 10.0, 10.5)-projection(theta, phi, ball_x, ball_y))

            # Additive noise for endogenous vars (if any)
            if noise_type == 'normal':
                shade = np.random.normal(shade, noise_level)
                mid = np.random.normal(mid, noise_level)
            elif noise_type == 'uniform':
                shade = np.random.uniform(shade-noise_level, shade+noise_level)
                mid = np.random.uniform(mid-noise_level, mid+noise_level)

            df_labels.append([i,j,shade,mid])

    # Save tabular data
    df_labels = pd.DataFrame(df_labels,columns=['i', 'j', 'shade','mid'])
    df_labels.to_csv(f'./causal_data/pendulum_{file_name}.csv',index=False)
    print(f'Data generated as ./causal_data/pendulum_{file_name}.csv')

    return df_labels


class torch_dataloader(data.Dataset):

    def __init__(self, data_array):
        self.data = data_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_ID_splt(df,train_frac=0.75,split=True,batch_size=32,shuffle=True):

    if split:
        # Load tabular data
        train=df.sample(frac=train_frac)
        test=df.drop(train.index)

        train_set = torch_dataloader(train.to_numpy())
        test_set = torch_dataloader(test.to_numpy())
        
        return data.DataLoader(test_set, batch_size=len(test_set), shuffle=shuffle), data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    else:
        return data.DataLoader(torch_dataloader(df.to_numpy()), batch_size=batch_size, shuffle=shuffle)

def load_OOD_split(df,column,quantile=0.75,batch_size=32, shuffle=True):
    if quantile > 0.5:
        test = df[df[column] > df[column].quantile(quantile)]
        train = df.drop(test.index)
    else:
        train = df[df[column] > df[column].quantile(quantile)]
        test = df.drop(train.index)

    train_set = torch_dataloader(train.to_numpy())
    test_set = torch_dataloader(test.to_numpy())
    
    return data.DataLoader(test_set, batch_size=len(test_set), shuffle=shuffle), data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)


def plot_w_err(data,title,xlabel,ylabel,var=False,sample=True,alpha=0.1,hline=False,label_append=None,hline_label = 'Target',axes=None,filt_low=0):
    '''
    Description: Plots a dictionary of arrays as means with std or variance error cones

    Inputs:
        data: a dictionary of array of the form [x,y,yerr]
        title: title of the plot
        xlabel: x-axis label
        ylabel: y-axis label
        var: if True, plot the variance instead of the standard deviation
        sample: if True, use sample mean/var (n-1)
        alpha: transparency of the error bars
        hline: if True, plot a horizontal line at the target value
        hline_label: label for the horizontal line
    '''

    if sample: ddof = 1
    else: ddof = 0

    for key in data.keys():

        if filt_low:
            ind = np.argsort(data[key][:,-1])[0:filt_low]
            data[key] = data[key][ind,:]

        mean = np.nanmean(data[key],axis=0)
        if label_append is not None:
            label = key + label_append
        else:
            label = key
        if axes !=None:
            p = plt.plot(mean, label=label,axes=axes)
        else:
            p = plt.plot(mean, label=label)
        if var: err = np.nanvar(data[key],axis=0,ddof=ddof)/data[key].shape[0]
        else: err = np.nanstd(data[key],axis=0,ddof=ddof)/data[key].shape[0]

        if axes !=None:
            plt.fill_between(np.arange(len(mean)), mean-err, mean+err, alpha=alpha,color=p[0].get_color(),axes=axes)
            if hline != False: plt.hlines(hline, 0, mean.shape[0], color ='r',linestyles='dashed',label=hline_label,axes=axes)
        else:
            plt.fill_between(np.arange(len(mean)), mean - err, mean + err, color=p[0].get_color(), alpha=alpha)
            if hline != False: plt.hlines(hline, 0, mean.shape[0], color ='r',linestyles='dashed',label=hline_label)

    if axes !=None:
        plt.xlabel(xlabel, fontsize=13,fontweight='bold',axes=axes)
        plt.ylabel(ylabel, fontsize=13,fontweight='bold',axes=axes)
        plt.title(title, fontsize=16,fontweight='bold',axes=axes)
        plt.ticklabel_format(axis='y', style='plain',axes=axes)
        plt.legend(axes=axes)
    else:
        plt.xlabel(xlabel, fontsize=13,fontweight='bold')
        plt.ylabel(ylabel, fontsize=13,fontweight='bold')
        plt.title(title, fontsize=16,fontweight='bold')
        plt.ticklabel_format(axis='y', style='plain')
        plt.legend()    


def plot_median(data,title,xlabel,ylabel,alpha=0.1,label_append=None,hline_label = 'Target',axes=None,filt_low=0):
    '''
    Description: Plots a dictionary of arrays as means with std or variance error cones

    Inputs:
        data: a dictionary of array of the form [x,y,yerr]
        title: title of the plot
        xlabel: x-axis label
        ylabel: y-axis label
        var: if True, plot the variance instead of the standard deviation
        sample: if True, use sample mean/var (n-1)
        alpha: transparency of the error bars
        hline: if True, plot a horizontal line at the target value
        hline_label: label for the horizontal line
    '''


    for key in data.keys():
        if filt_low:
            ind = np.argsort(data[key][:,-1])[0:filt_low]
            data[key] = data[key][ind,:]


        median = np.nanmedian(data[key],axis=0)
        if label_append is not None:
            label = key + label_append
        else:
            label = key
        if axes !=None:
            p = plt.plot(median, label=label,axes=axes)
        else:
            p = plt.plot(median, label=label)

        if axes !=None:
            for row in range(data[key].shape[0]):
                plt.plot(data[key][row,:], color=p[0].get_color(),alpha=alpha,axes=axes)
        else:
            for row in range(data[key].shape[0]):
                plt.plot(data[key][row,:], color=p[0].get_color(),alpha=alpha)

    if axes !=None:
        plt.xlabel(xlabel, fontsize=13,fontweight='bold',axes=axes)
        plt.ylabel(ylabel, fontsize=13,fontweight='bold',axes=axes)
        plt.title(title, fontsize=16,fontweight='bold',axes=axes)
        plt.ticklabel_format(axis='y', style='plain',axes=axes)
        plt.legend(axes=axes)
    else:
        plt.xlabel(xlabel, fontsize=13,fontweight='bold')
        plt.ylabel(ylabel, fontsize=13,fontweight='bold')
        plt.title(title, fontsize=16,fontweight='bold')
        plt.ticklabel_format(axis='y', style='plain')
        plt.legend()    


class signSGD(optim.Optimizer):

    def __init__(self, params, lr=0.01, rand_zero=True):
        defaults = dict(lr=lr)
        self.rand_zero = rand_zero
        super(signSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # take sign of gradient
                grad = torch.sign(p.grad)

                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
                    assert not (grad==0).any()
                
                # make update
                p.data -= group['lr'] * grad

        return loss

