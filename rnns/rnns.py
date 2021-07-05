# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:44:16 2021

@author: Estefany Suarez
"""

import torch
from torch import nn

class NeuralNet(nn.Module):

    def __init__(self, input_size=None, output_size=None, hidden_size=None,
                 bias=False, n_layers=1, nonlinearity='tanh', h0=None,
                 init_input=None, init_hidden=None, init_output=None,
                 device='cpu'):

        """
            Constructor class for RNNs

            Parameters
            ----------
            input_size : int
                Number of neurons in the input layer. Equal to the number of
                input features.

            output_size : int
                Number of neurons in the output layer. Equal to the number of
                output features.

            hidden_size : int
                Number of neurons in the hidden layer.

            n_layers : int
                Default 1, number of hidden layers

            nonlinearity : str {'tanh', 'relu'}, default 'tanh'
                Activation function

            init_input  : str {'', ''}, default None
                Initialization of input layer's weights. If None, Pytorch's
                default weight initialization will be used.

            init_hidden : str {'', ''}, default None
                Initialization of hidden layer's weights. If None, Pytorch's
                default weight initialization will be used.

            init_output : str {'', ''}, default None
                Initialization of output layer's weights.If None, Pytorch's
                default weight initialization will be used.

            device : str {'cpu', 'gpu'}, default 'cpu'
                Device to use for training.
        """
        super(NeuralNet, self).__init__()

        # defining model parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.n_layers = n_layers
        self.device = device

        # defining model layers
        # RNN Layer
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          nonlinearity=nonlinearity,
                          bias=bias,
                          batch_first=True,
                          # 0.0, #dropout=0,
                          bidirectional=False
                          )

        # Fully connected layer
        self.readout = nn.Linear(in_features=hidden_size,
                                 out_features=output_size,
                                 bias=False
                                 )

        # weights initialization
        if h0 is not None:
            self.set_rnn_weights(h0)

        # input weights initialization - Wih
        if init_input is not None:
            self.init_input_weights(init_input, nonlinearity)

        # hidden weights initialization - Whh
        if init_hidden is not None:
            self.init_hidden_weights(init_hidden, nonlinearity)

        # output weights initialization - Who
        if init_output is not None:
            self.init_output_weights(init_output, nonlinearity)


    # methods
    def init_weights(self, w, init_method, nonlinearity):

        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(w,
                                    # gain=nn.init.calculate_gain(nonlinearity)
                                    )

        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(w,
                                   # gain=nn.init.calculate_gain(nonlinearity)
                                   )

        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w,
                                     # nonlinearity=nonlinearity
                                     )

        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(w,
                                    # nonlinearity=nonlinearity
                                    )

        elif init_method == 'ones':
            nn.init.ones_(w)


    def init_input_weights(self, init_method, nonlinearity):

        # initialize tensor
        w_ih = torch.empty(self.hidden_size, self.input_size)
        w_ih = w_ih.to(self.device)

        # draw weights according to init_method
        self.init_weights(w_ih, init_method, nonlinearity)

        # assign new weights
        self.rnn.weight_ih_l0 = torch.nn.Parameter(w_ih, requires_grad=False)


    def init_hidden_weights(self, init_method, nonlinearity):

        # initialize tensor
        w_hh = torch.empty(self.hidden_size, self.hidden_size)
        w_hh = w_hh.to(self.device)

        # draw weights according to init_method
        self.init_weights(w_hh, init_method, nonlinearity)

        # assign new weights
        self.rnn.weight_hh_l0 = torch.nn.Parameter(w_hh, requires_grad=True)


    def init_output_weights(self, init_method, nonlinearity):

        # initialize tensor
        w_ho = torch.empty(self.output_size, self.hidden_size)
        w_ho = w_ho.to(self.device)

        # draw weights according to init_method
        self.init_weights(w_ho, init_method, nonlinearity)

        # assign new weights
        self.readout.weight = torch.nn.Parameter(w_ho, requires_grad=True)


    def init_hidden_state(self, n_iters):
        # Initialize hidden state for the forward pass. This initialization
        # takes place at every batch.
        hidden = torch.zeros(self.n_layers,
                             n_iters,
                             self.hidden_size
                             )

        hidden = hidden.to(self.device)

        return hidden.type(torch.float)


    def forward(self, x, hidden=None):

        n_iters = x.size(0)

        # Initialize hidden state
        if hidden is None:
            hidden = self.init_hidden_state(n_iters)

        # Pass in input and initial hidden state into the rnnmodel
        # and obtain outputs
        out, hidden = self.rnn(x, hidden)

        # Reshape outputs to be fit into the readout module
        out = out.contiguous().view(-1, self.hidden_size)

        # Pass rnn outputs into the readout module
        out = self.readout(out)

        out = out.to(self.device)
        hidden = hidden.to(self.device)

        return out, hidden


    def get_rnn_weights(self, i=1, layer=-1):
        return self.rnn.all_weights[layer][i].detach().numpy().copy()


    def get_output_weights(self):
        return self.readout.weight.detach().numpy().copy()


    def set_rnn_weights(self, w, update=True):
        w = torch.tensor(w, dtype=torch.float, device=self.device)
        self.rnn.weight_hh_l0 = torch.nn.Parameter(w, requires_grad=update)


    def apply_sparse_structure(self, sparse_w, update=True):
        new_w = sparse_w * self.rnn.weight_hh_l0.detach().numpy().copy()
        new_w = torch.tensor(new_w, dtype=torch.float, device=self.device)
        self.rnn.weight_hh_l0 = torch.nn.Parameter(new_w, requires_grad=update)


class BioNeuralNet(NeuralNet):
    def __init__(self, win=None, w=None, remap_w=False, init_method=None, *args, **kwargs):
        """
            Constructor class for RNNs

            Parameters
            ----------
            input_size : int, number of input neurons
            output_size : int, number of output neurons
            hidden_size : int, number of neurons in the hidden layer
            n_layers : int, default 1, number of hidden layers
            nonlinearity : str {'tanh', 'relu'}, default 'tanh', activation function
            init_input  : str {'', ''}, default 'default', initialization of input layer's weights
            init_hidden : str {'', ''}, default 'default', initialization of hidden layer's weights
            init_output : str {'', ''}, default 'default', initialization of output layer's weights
            device : str {'cpu', 'gpu'}, default 'cpu'
            win : torch.Tensor of shape (hidden_size,input_size) denoting the topology (weighted or
            binary) of the hidden recurrent layer.
            w : torch.Tensor of shape (hidden_size,hidden_size) denoting the topology (weighted or
            binary) of the hidden recurrent layer.
            remap_w : bool, default False. If True, a new distribution of weights is generated according to
            init-method, and is assigned based on the connection ranking of the passed matrix w. If False,
            w is used to redefine the connectivity of the RNN.
            init_method : str {'', ''}, default is None. Initialization method to generate new remapped
            weights. Only matters if remap_w is True. If None, default Pytorch's initialization
            method is used.
        """
        super(BioNeuralNet, self).__init__(*args, **kwargs)
        self.set_weights(win, w, init_method, remap_w)

    def set_weights(self, win, w, init_method, remap_w):
        if win is not None:
            w_ih = torch.tensor(win, dtype=torch.float)
            self.rnn.weight_ih_l0 = torch.nn.Parameter(w_ih, requires_grad=False)

        if w is not None:
            if remap_w:
                w_hh = self.remap_weights(w, init_method)
            else:
                w_hh = torch.tensor(w, dtype=torch.float)
            self.rnn.weight_hh_l0 = torch.nn.Parameter(w_hh, requires_grad=True)


    def remap_weights(self, w, init_method):

        print("Remapping weights ...")

        # convert connectivity matrix to tensor
        w = torch.tensor(w)

        # get nonzero connectivity values
        idx = torch.where(w != 0)
        w_vals = w[idx]

        # sort connectivity values
        sort_w_vals, _ = torch.sort(w_vals)

        # rank connectivity values
        rank = torch.tensor([torch.where(sort_w_vals == v)[0][0] for v in w_vals])

        # create new connectivity values according to initialization method
        if init_method is None:
            new_w_vals = self.rnn.weight_hh_l0[idx]
        else:
            new_w_vals = self.init_weights(len(w_vals), init_method)

        # sort new connectivity
        sort_new_w_vals, _ = torch.sort(new_w_vals)

        # create empty tensor for new connectivity matrix
        new_w = torch.zeros_like(w, dtype=sort_new_w_vals.dtype)

        # remap new values
        new_w[idx] = torch.squeeze(sort_new_w_vals)[rank]

        return new_w.type(torch.float)


    def init_weights(self, size, init_method):
        w = torch.empty(1,size)

        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(w)

        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(w)

        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w)

        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(w)

        return w


class Reservoir(nn.Module):

    def __init__(self, input_size=None, hidden_size=None,
                 nonlinearity='tanh', w_ih=None, w_hh=None,
                 device='cpu'):

        """
            Constructor class for RNNs

            Parameters
            ----------
            input_size : int
                Number of neurons in the input layer. Equal to the number of
                input features.

            output_size : int
                Number of neurons in the output layer. Equal to the number of
                output features.

            hidden_size : int
                Number of neurons in the hidden layer.

            n_layers : int
                Default 1, number of hidden layers

            nonlinearity : str {'tanh', 'relu'}, default 'tanh'
                Activation function

            init_input  : str {'', ''}, default None
                Initialization of input layer's weights. If None, Pytorch's
                default weight initialization will be used.

            init_hidden : str {'', ''}, default None
                Initialization of hidden layer's weights. If None, Pytorch's
                default weight initialization will be used.

            init_output : str {'', ''}, default None
                Initialization of output layer's weights.If None, Pytorch's
                default weight initialization will be used.

            device : str {'cpu', 'gpu'}, default 'cpu'
                Device to use for training.
        """
        super(Reservoir, self).__init__()

        # defining model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = 1
        self.bias = False
        self.device = device

        # defining model layers
        # RNN Layer
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          nonlinearity=nonlinearity,
                          bias=False,
                          batch_first=True,
                          # 0.0, #dropout=0,
                          bidirectional=False
                          )

        # weights initialization
        if w_ih is not None:
            self.set_input_weights(w_ih)

        if w_hh is not None:
            self.set_rnn_weights(w_hh)


    # methods
    def init_hidden_state(self, n_iters):
        # Initialize hidden state for the forward pass. This initialization
        # takes place at every batch.
        hidden = torch.zeros(self.n_layers,
                             n_iters,
                             self.hidden_size
                             )

        hidden = hidden.to(self.device)

        return hidden.type(torch.float)


    def forward(self, x, hidden=None):

        n_iters = x.size(0)

        # Initialize hidden state
        if hidden is None: hidden = self.init_hidden_state(n_iters)

        # Pass in input and initial hidden state into the rnnmodel
        # and obtain outputs
        out, _ = self.rnn(x, hidden)
        out = out.to(self.device)

        return out


    def set_input_weights(self, w_ih, update=False):
        w_ih = torch.tensor(w_ih, dtype=torch.float, device=self.device)
        self.rnn.weight_ih_l0 = torch.nn.Parameter(w_ih, requires_grad=update)


    def get_input_weights(self):
        return self.rnn.all_weights[0][0].detach().numpy().copy()


    def set_rnn_weights(self, w_hh, update=False):
        w_hh = torch.tensor(w_hh, dtype=torch.float, device=self.device)
        self.rnn.weight_hh_l0 = torch.nn.Parameter(w_hh, requires_grad=update)


    def get_rnn_weights(self):
        return self.rnn.all_weights[0][1].detach().numpy().copy()


class LinearRegression(nn.Module):

    def __init__(self, input_size, output_size, bias=False, device='cpu'):
        super(LinearRegression, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = device

        self.linear = torch.nn.Linear(input_size,
                                      output_size,
                                      bias=False
                                      )

    def forward(self, x):
        # x = x.contiguous().view(-1, self.hidden_size)
        out = self.linear(x)
        out = out.to(self.device)
        return out
