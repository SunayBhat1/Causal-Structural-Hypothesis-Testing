import torch
import torch.nn as nn                          
import torch.nn.functional as F               
torch.set_default_dtype(torch.float32)

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Outline:
-VAE Class
-DAG Layer Class
-Image Encoder Class
-Image Decoder Class
-Label Encoder Class
-Label Decoder Class


Things we could add:
-Scale vector for labels, running unscaled right now
'''

###############################################################################


def soft_lrelu(x):
    # Reducing to ReLU when a=0.5 and e=0
    # Here, we set a-->0.5 from left and e-->0 from right,
    # where adding eps is to make the derivatives have better rounding behavior around 0.
    a = 0.49
    e = torch.finfo(torch.float32).eps
    return (1 - a) * x + a * torch.sqrt(x * x + e * e) - a * e

class slReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return soft_lrelu(input)

class energy_layer(nn.Module):
    def __init__(self):
        super(energy_layer, self).__init__()
    def forward(self, x):
        print(torch.square(x).mean().item())
        return x

class MLP(nn.Module):
    '''
    A fully connected NN in pytroch (Multi-Layer Perceptron).
    '''
    def __init__(self,
                  input_dim,
                  output_dim,
                  layers=None,
                  binary_out=None,
                  bias=True,
                  print_energy=False,
                  activation=slReLU(),
                  name=None,
                  ):

        """
        Initialize a new FullyConnectedNet.
        
        Inputs:
        - input_dim: An integer giving the size of the input.
        - output_dim: An integer giving the size of the output (classes or values to predict).
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if layers == None:
            self.layers = [32,32,32]
        else:
            self.layers = layers
        
        if binary_out == None:
            self.binary_out = torch.zeros(output_dim)
        else:
            self.binary_out = binary_out
            self.sigmoid = nn.Sigmoid()
        self.print_energy = print_energy
        
        self.bias = bias
        self.activation = activation

        self.input2hidden = nn.Linear(in_features=input_dim, out_features=self.layers[0],bias=bias)
        if print_energy: self.print_i2h = energy_layer()
        hidden_layers = []
        for i in range(len(self.layers)-1): 
            hidden_layers.append(nn.Linear(in_features=self.layers[i], out_features=self.layers[i+1],bias=bias))
            if print_energy: hidden_layers.append(energy_layer())
            hidden_layers.append(self.activation)
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.hidden2output = nn.Linear(in_features=self.layers[-1], out_features=output_dim,bias=bias)

    def forward(self, x):
        
        if self.print_energy: x = self.hidden_layers(self.activation(self.print_i2h(self.input2hidden(x))))
        else: x = self.hidden_layers(self.activation(self.input2hidden(x)))
        x = self.hidden2output(x)
        for i in range(len(self.binary_out)):
            if self.binary_out[i]:
                x[:,i] = self.sigmoid(x[:,i])
        return x

###############################################################################

class VAE(nn.Module):
    '''
    simple VAE
    '''
    def __init__(self, 
                data_dim, 
                latent_dim,
                binary_out = None,
                layers=[128, 128, 128],
                name=None,
                print_energy=False,
                ):
        super().__init__()
        '''
        Inputs:
        -data_dim: dimension of data
        -latent_dim: dimension of latent space
        -width (default: 128): width of end/dec hidden layers
        -depth (default: 3): depth of end/dec hidden layers
        -name (default: None): name of model
        '''

        # Class Variables
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.binary_out = binary_out
        self.layers = layers
        
        self.name = name

        if binary_out == None:
            self.binary_out = torch.zeros(data_dim)
        else:
            self.binary_out = binary_out

        self.enc_mean = MLP(self.data_dim, self.latent_dim, self.layers)
        self.enc_var = MLP(self.data_dim, self.latent_dim, self.layers)

        self.dec = MLP(self.latent_dim, self.data_dim, self.layers[::-1],binary_out=binary_out,print_energy=print_energy)


    def forward(self, x_in):

        means = self.enc_mean(x_in)
        logvars = self.enc_var(x_in)

        if self.training:
            z = torch.randn_like(means, device=device).mul(
                torch.exp(0.5 * logvars)).add_(means)
        else:
            z = means

        reconstruction = self.dec(z)

        return reconstruction, means, logvars, z

###############################################################################

## CausalGen
class CGen(nn.Module):
    '''
    Causal Gen Model
    '''
    def __init__(self,scm,
                 theta_layers,
                 binary_out=None,
                 dropout_rate=0,
                 print_energy=False,
                 ):
        '''
        Inputs:
            -scm: structural causal model
            -causal_layers: list of hidden layer sizes for causal theta nets
            -binary_out: list of binary outputs
            -dropout_rate: dropout rate for dropout net
        '''
        super().__init__()

        self.n_features = scm.shape[0]
        self.theta_layers = theta_layers
        self.name = None

        if binary_out == None:
            self.binary_out = torch.zeros(self.n_features)
        else:
            self.binary_out = binary_out

        self.dag = nn.Parameter(torch.from_numpy(scm - np.diag(np.diag(scm))),requires_grad=False)
        self.d = nn.Parameter(torch.from_numpy(np.array(scm.diagonal())), requires_grad=False)

        self.theta_nets = nn.ModuleList([MLP(self.n_features,1,layers=theta_layers,binary_out=[self.binary_out[i]],print_energy=print_energy) for i in range(self.n_features)])

        self.dropout_net = nn.Dropout(p=dropout_rate)

    @property
    def SCM(self):
        return self.dag + torch.diag(self.d) 

    def forward(self, x):

        x_hadamard = torch.einsum('ln,bl->bln', self.SCM, x).type(torch.float32)
        x_recon = torch.zeros(x_hadamard.shape[0],
                                self.n_features,
                                device=device)

        for i in range(self.n_features):

            # Dropout inputs into g_nets
            if torch.sum(self.SCM[:, i]) > 1:
                x_recon[:, i] = torch.squeeze(self.theta_nets[i](self.dropout_net(
                    x_hadamard[:, :, i])))
            else:
                x_recon[:, i] = torch.squeeze(self.theta_nets[i](x_hadamard[:, :, i]))
                
        return x_recon


class CVGen(nn.Module):
    '''
    Causal Variational Gen Model
    '''
    def __init__(self, scm, 
                 enc_layers, 
                 theta_layers, 
                 binary_out=None, 
                 dropout_rate = 0, 
                 name=None,
                 print_energy=False,
                 ):
        super().__init__()
        '''
        Inputs:
            -scm: structural causal model
            -enc_layers: list of hidden layer sizes for encoder/decoder
            -causal_layers: list of hidden layer sizes for causal theta nets
            -binary_out: list of binary outputs
            -dropout_rate: dropout rate for dropout net
            -name: name of model
        '''

        # Class Variables
        self.scm = scm
        self.n_features = scm.shape[0]
        self.enc_layers = enc_layers
        self.theta_layers = theta_layers
        self.dropout_rate = dropout_rate
        self.name = name

        if binary_out == None:
            self.binary_out = torch.zeros(self.n_features)
        else:
            self.binary_out = binary_out

        ## Enc/Dec
        self.mean_enc = nn.ModuleList([MLP(1, 1, self.enc_layers,print_energy=print_energy) for _ in range(self.n_features)])
        self.var_enc = nn.ModuleList([MLP(1, 1, self.enc_layers,print_energy=print_energy) for _ in range(self.n_features)])
        self.dec = nn.ModuleList([MLP(1, 1, self.enc_layers[::-1],print_energy=print_energy) for _ in range(self.n_features)])

        # Causal Layer
        self.CGen = CGen(self.scm, 
                        theta_layers=self.theta_layers,
                        binary_out= self.binary_out,
                        dropout_rate=dropout_rate,
                        )

    def forward(self, x):

        means = torch.cat(
            [self.mean_enc[i](x[:, i:i + 1]) for i in range(self.n_features)]
            ,dim=1)
        logvars = torch.cat(
            [self.var_enc[i](x[:, i:i + 1]) for i in range(self.n_features)],
            dim=1)

        # Sampling: Sample z from latent space using mu and logvar
        if self.training:
            z = torch.randn_like(means, device=device).mul(
                torch.exp(0.5 * logvars)).add_(means)
        else:
            z = means

        # Causal Layer
        z_recon = self.CGen(z)

        # Decoding
        label_recon = torch.cat(
            [self.dec[i](z_recon[:, i:i + 1]) for i in range(self.n_features)],
            dim=1)

        return label_recon, means, logvars, z, z_recon
