#Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software;
#you can redistribute it and/or modify
#it under the terms of the MIT License.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
import torch.utils.data as Data
import torch.optim as optim 
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from adabelief_pytorch import AdaBelief
import igraph as ig
import numpy as np

from models import CGen,CVGen,VAE, MLP
import utils as ut
import data_utils as data_ut
import dag_utils as dag_ut
import preconditioned_stochastic_gradient_descent as psgd

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# Read in processed csv for pytorch dataloader
class MyDataset(Dataset):
    def __init__(self, root):
        self.df = pd.read_csv(root)
        self.data = self.df.to_numpy()
        self.x = torch.from_numpy(self.data[:,:])
    def __getitem__(self, idx):
        return self.x[idx, :]
    def __len__(self):
        return len(self.data)

# Dataloader for tabular data
def load_data_batch(root,batch_size=32,shuffle=True):
    myData = MyDataset(root)
    return DataLoader(myData, batch_size=batch_size, shuffle=shuffle)


def matrix_poly(matrix, d):
    x = torch.eye(d).to(device) + torch.div(matrix.to(device), d).to(device)
    return torch.matrix_power(x, d)

# DAG loss, see DAG No TEARS paper
def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

class dataload_withlabel(Data.Dataset):
    def __init__(self, root, dataset="train"):
        root = root + "/" + dataset

        imgs = os.listdir(root)

        self.dataset = dataset

        self.imgs = [os.path.join(root, k) for k in imgs]
        for k in imgs:
            try:
                list(map(int, k[:-4].split("_")[1:]))
            except:
                print(k)
        self.imglabel = [list(map(int, k[:-4].split("_")[1:])) for k in imgs]
        # print(self.imglabel)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        #print(idx)
        img_path = self.imgs[idx]

        label = torch.from_numpy(np.asarray(self.imglabel[idx]))
        #print(len(label))
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        array1 = np.asarray(label)
        label = torch.from_numpy(array1)
        data = torch.from_numpy(array)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img).reshape(96, 96, 4)
            data = torch.from_numpy(pil_img)

        return data, label.float()

    def __len__(self):
        return len(self.imgs)

def get_batch_unin_dataset_withlabel(dataset_dir,
                                     batch_size,
                                     dataset="train",
                                     shuffle=True):

    dataset = dataload_withlabel(dataset_dir, dataset)
    dataset = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset


def save_model_by_name(model, epochs):

    save_dir = os.path.join('/Users/sunaybhat/Documents/GitHub/Causal_Deep_Learning/models', model.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, 'model-{:05d}.pt'.format(epochs))
    state = model.state_dict()
    torch.save(state, file_path)
    print('Saved to {}'.format(file_path))

def sample_model(model,df,num_points,use_causal=True,columns=None):
    model.train()
    z = torch.from_numpy(df[0:num_points].to_numpy()).type(torch.float32).to(device)
    x_recon,_,_,_,_,_ = model(z,use_causal)
    return pd.DataFrame(x_recon.detach().cpu().numpy(),columns=columns)

def sample_model_causal(model,df,num_points,columns=None):
    z = torch.from_numpy(df[0:num_points].to_numpy()).type(torch.float32).to(device)
    x_recon, _, _ = model(z)
    return pd.DataFrame(x_recon.detach().cpu().numpy(),columns=columns)


def load_model_by_name(model, global_step):
  """
	Load a model based on its name model.name and the checkpoint iteration step

	Args:
		model: Model: (): A model
		global_step: int: (): Checkpoint iteration
	"""
  file_path = os.path.join('/Users/sunaybhat/Documents/GitHub/Causal_Deep_Learning/models', model.name,
                           'model-{:05d}.pt'.format(global_step))
  state = torch.load(file_path)
  model.load_state_dict(state)
#   print("Loaded from {}".format(file_path))

def test(model, test_loader,mode='CGen'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.type(torch.float32)

            if mode == 'CGen': recon_batch = model(data)
            elif mode == 'CVGen': recon_batch, mu, logvar,_,_ = model(data)
            elif mode == 'VAE': recon_batch,mu,logvar,_ = model(data)
            elif mode== 'NN': recon_batch = model(data[:,:-1])

            test_loss += F.mse_loss(recon_batch, data).item()
    return test_loss,data.detach().numpy(), recon_batch.detach().numpy()

def train_epoch(model,optimizer,train_data,model_type,device,loss_type='MSE'):

    losses = 0
    for x_batch in train_data:
        model.train()
        optimizer.zero_grad()

        if model_type == 'CGen':
            x_hat = model(x_batch.type(torch.float32).to(device))
        elif model_type == 'CVGen':
            x_hat, mu, logvar,_,_ = model(x_batch.type(torch.float32).to(device))

        if loss_type == 'MSE':
            loss = F.mse_loss(x_hat, x_batch.type(torch.float32).to(device))

        losses += loss.item()
        loss.backward()
        optimizer.step()

    losses /= len(train_data)

    return losses


def train_epoch_PSGD(model,optimizer,train_data,model_type,device,loss_type='MSE'):

    losses = 0
    for x_batch in train_data:
        model.train()
    
        def closure():
            if model_type == 'CGen':
                x_hat = model(x_batch.type(torch.float32).to(device))
            elif model_type == 'CVGen':
                x_hat, mu, logvar,_,_ = model(x_batch.type(torch.float32).to(device))

            if loss_type == 'MSE':
                loss = F.mse_loss(x_hat, x_batch.type(torch.float32).to(device))            
            return loss 

        loss = optimizer.step(closure)
        losses += loss.item()
        

    losses /= len(train_data)

    return losses


def get_model_optimizer(SCM,model_name,len_train,optim_type = 'AdaBelief',mode='linear'):
  if model_name == 'CGen': 
    if mode == 'linear':
      model = CGen(SCM,theta_layers = [4])
    else:
      model = CGen(SCM,theta_layers = [100,100])

  if model_name == 'CVGen': 
    if mode == 'linear':
      model = CVGen(SCM,enc_layers = [4],theta_layers = [4,4])
    else:
      model = CVGen(SCM,enc_layers = [4],theta_layers = [4,16,8,2])
  
  if optim_type == 'AdaBelief':
      optimizer = AdaBelief(
          model.parameters(),
          lr=1e-3, 
          eps=1e-16, 
          betas=(0.9,0.999), 
          weight_decouple = False, 
          rectify = False,
          print_change_log = False,
          )
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=len_train,
                                                        eta_min=0,
                                                         last_epoch=-1)
  elif optim_type == 'PSGD':
      optimizer = psgd.UVd(
          model.parameters(),
          rank_of_approximation=100,
          lr_params = 0.05,
          lr_preconditioner=0.2,
          momentum = 0.9,
          # grad_clip_max_norm = 10,
          preconditioner_update_probability=0.9,
      )
      shed = optim.SGD(model.parameters(),lr=0.1)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(shed,
                                                          T_max=len_train,
                                                          eta_min=0,
                                                          last_epoch=-1)

  
  return model, optimizer, scheduler