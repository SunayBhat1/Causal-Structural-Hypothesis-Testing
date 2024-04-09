import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch
from torch.nn import functional as F       
import torch.optim as optim 
from torchvision import datasets, transforms
from torch.utils import data
from adabelief_pytorch import AdaBelief
import igraph as ig
import numpy as np

from models import CGen,CVGen,VAE, MLP
import utils as ut
import data_utils as data_ut
import dag_utils as dag_ut
import preconditioned_stochastic_gradient_descent as psgd

def train_dataset(DAG_true,args,train_data,test_data,device):
    losses = {}
    losses['train'] = {}
    losses['test'] = {}

    if len(train_data) <= len(test_data):
      raise Exception("Train Dataset ({}) smaller than test dataset ({}), Check data splits".format(len(train_data),len(test_data)))

    pbar = tqdm(args.hamm_distances)
    for i_hamm,hamm_dist in enumerate(pbar):
        losses['train'][hamm_dist] = np.zeros((args.mod_iters,args.epochs))
        losses['test'][hamm_dist] = np.zeros((args.mod_iters,args.epochs))

        for iter in range(args.mod_iters):

            DAG_mod, SHDN, SHDP = dag_ut.modify_DAG(DAG_true, hamm_dist)

            losses['test'][str(hamm_dist)+'_SHDN_'+str(iter)] = SHDN
            losses['test'][str(hamm_dist)+'_SHDP_'+str(iter)] = SHDP

            # Add diags for exo vars
            SCM = np.diag(DAG_true.sum(axis=0) == 0) + DAG_mod

            model,optimizer,scheduler = ut.get_model_optimizer(SCM,args.model_type,len(train_data),args.optim_type,args.dag_type)
            
            model = model.to(device)

            for epoch in range(args.epochs):

                if args.optim_type == 'PSGD':
                    losses['train'][hamm_dist][iter,epoch] = ut.train_epoch_PSGD(model,optimizer,train_data,args.model_type,device)
                else:
                    losses['train'][hamm_dist][iter,epoch] = ut.train_epoch(model,optimizer,train_data,args.model_type,device)

                losses['test'][hamm_dist][iter,epoch],_,_ = ut.test(model,test_data,mode=args.model_type)

                scheduler.step()
                if args.optim_type == 'PSGD':
                    optimizer.lr_params = scheduler.get_last_lr()[-1]

                if args.verbose and (epoch % 10 == 0):
                    pbar.set_description(f'Hamm: {i_hamm}/{len(args.hamm_distances)}, ' + 
                                        f'Iter: {iter}/{args.mod_iters}, ' +
                                        f'Epoch: {epoch}/{args.epochs}, ' + 
                                        f'Train Loss: {losses["train"][hamm_dist][iter,epoch]:.4f}, ' +
                                        f'Test Loss: {losses["test"][hamm_dist][iter,epoch]:.4f}')

    return losses

def main():
    '''
    '''

    # Parse command line input arguments and defaults
    parser=argparse.ArgumentParser(description='Train Forward Model')
    parser.add_argument('--graph_size',type=int,default=4,help='Number of nodes in the graph',)
    parser.add_argument('--graph_edges',type=int,default=4,help='Number of edges in the graph',)
    parser.add_argument('--dag_type',type=str,default='linear',help='linear or non-linear',)
    parser.add_argument('--dag_iters',type=int,default=5,help='Number of data samples generated',)
    parser.add_argument('--N',type=int,default=100,help='Number of data samples generated',)
    parser.add_argument('--hamm_distances', type=int, nargs="*", default=[0,1,2,3,4],help='Hamming distances to test',)
    parser.add_argument('--mod_iters',type=int,default=5,help='Iterations of models',)
    parser.add_argument('--epochs',type=int,default=100,help='Epochs per model',)
    parser.add_argument('--random_seed',type=int,default=1,help='Random See (0 if none)',)
    parser.add_argument('--model_type',type=str,default='CGen',help='CGen or CVGen',)
    parser.add_argument('--optim_type',type=str,default='PSGD',help='Optimizer',)
    parser.add_argument('--noise',type=str,default='gauss',help='Type of Noise for linear',)
    parser.add_argument('--nonlinear_type',type=str,default='mlp',help='Type of Nonlinear gen',)
    parser.add_argument('--no_verbose',action='store_true',help='Print Stuff',)

    args=parser.parse_args()

    if args.no_verbose:
        args.verbose = False
    else:
        args.verbose = True

    plt.style.use('seaborn')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.random_seed != 0: dag_ut.set_random_seed(1)

    for iter in range(args.dag_iters):
        print(f'\n{iter}/{args.dag_iters} DAG Iters\n')
          
        # Sim DAG
        DAG_true = dag_ut.simulate_dag(args.graph_size, args.graph_edges)
    
        for args.snr in [float('inf'),5]:
    
            # Model Save folder
            if not os.path.isdir('results'):
                os.mkdir('results')
            if not os.path.isdir(f'results/{args.graph_size}_{args.graph_edges}'):
                  os.mkdir(f'results/{args.graph_size}_{args.graph_edges}')
            if not os.path.isdir(f'results/{args.graph_size}_{args.graph_edges}/{iter}_{args.dag_type}_{args.model_type}_{args.snr}'):
                os.mkdir(f'results/{args.graph_size}_{args.graph_edges}/{iter}_{args.dag_type}_{args.model_type}_{args.snr}')
            save_dir = f'results/{args.graph_size}_{args.graph_edges}/{iter}_{args.dag_type}_{args.model_type}_{args.snr}'
            run_dir = f'results/{args.graph_size}_{args.graph_edges}'
    
            # Sim Paramters
            if args.dag_type == 'linear':
              W_true = dag_ut.simulate_parameter(DAG_true)
              X = dag_ut.simulate_linear_sem(W_true, args.N,args.snr)
            elif args.dag_type == 'nonlinear':
              X = dag_ut.simulate_nonlinear_sem(DAG_true, args.N, args.nonlinear_type,args.snr)
    
            # In Distribution
            print('Runing ID')
            losses_ID = {}
            test_data,train_data = data_ut.load_ID_splt(pd.DataFrame(X).astype(float),split=True) 
            losses_ID = train_dataset(DAG_true,args,train_data,test_data,device)
            pickle.dump(losses_ID,open(f'{save_dir}/losses_ID.pkl','wb'))
    
            # Out of Distribution
            print('Runing ODs')
            losses_OD = {}
            for i in range(args.graph_size):
              test_data,train_data = data_ut.load_OOD_split(pd.DataFrame(X).astype(float),i)
              losses_OD[i] = train_dataset(DAG_true,args,train_data,test_data,device)
            pickle.dump(losses_OD,open(f'{save_dir}/losses_OD.pkl','wb'))
    
        # Save DAG
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(13,5))
        plt.subplot(1,2,1)
        sns.heatmap(DAG_true,annot=True)
        # plt.subplot(1,2,2)
        ig.plot(ig.Graph.Adjacency(DAG_true.tolist()),vertex_label=range(DAG_true.shape[0]),edge_color='black',layout='kk',bbox=(0,0,500,500),target=ax2)
        fig.savefig(os.path.join(run_dir,f'{iter}_{args.dag_type}_DAG_true.png'))
        
    with open(os.path.join(run_dir,f'{args.dag_type}_args.txt'),'w') as f:
        for key, value in args.__dict__.items():
            f.write('%s:%s\n' % (key, value))


if __name__=='__main__':
    main()
