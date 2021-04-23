import torch
from torch import nn

class NeuralNet(nn.Module):

    def __init__(self, input_size=None, output_size=None, hidden_size=None,
                 n_layers=1, nonlinearity='tanh',
                 init_input=None, init_hidden=None, init_output=None,
                 device='cpu'):

        """
            Constructor class for RNNs

            Parameters
            ----------
            input_size : int, number of input neurons
            output_size : int, number of output neurons
            hidden_size : int, number of neurons in the hidden layer
            n_layers : int, default 1, number of hidden layers
            nonlinearity : str {'tanh', 'relu'}, default 'tanh', activation function
            init_input  : str {'', ''}, default None, initialization of input layer's weights.
            If None, Pytorch's default weight initialization will be used.
            init_hidden : str {'', ''}, default None, initialization of hidden layer's weights.
            If None, Pytorch's default weight initialization will be used.
            init_output : str {'', ''}, default None, initialization of output layer's weights.
            If None, Pytorch's default weight initialization will be used.
            device : str {'cpu', 'gpu'}, default 'cpu'
        """
        super(NeuralNet, self).__init__()

        # defining model parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = device

        # defining model layers
        # RNN Layer
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          nonlinearity=nonlinearity,
                          bias=False,
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
        # input weights initialization - Wih
        if init_input is not None:
            self.init_input_weights(init_input)

        # hidden weights initialization - Whh
        if init_hidden is not None:
            self.init_hidden_weights(init_hidden)

        # output weights initialization - Who
        if init_output is not None:
            self.init_output_weights(init_output)


    # methods
    def init_input_weights(self, init_method):
        w_ih = torch.empty(self.hidden_size, self.input_size)
        # w_ih = w_ih.to(self.device)

        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(w_ih)

        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(w_ih)

        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w_ih)

        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(w_ih)

        self.rnn.weight_ih_l0 = torch.nn.Parameter(w_ih, requires_grad=True)


    def init_hidden_weights(self, init_method):
        w_hh = torch.empty(self.hidden_size, self.hidden_size)
        # w_hh = w_hh.to(self.device)

        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(w_hh)

        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(w_hh)

        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w_hh)

        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(w_hh)

        self.rnn.weight_hh_l0 = torch.nn.Parameter(w_hh, requires_grad=True)


    def init_output_weights(self, init_method):
        w_ho = torch.empty(self.output_size, self.hidden_size)
        # w_ho = w_ho.to(self.device)

        if init_method == 'xavier_uniform':
            nn.init.xavier_uniform_(w_ho)

        elif init_method == 'xavier_normal':
            nn.init.xavier_normal_(w_ho)

        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(w_ho)

        elif init_method == 'kaiming_normal':
            nn.init.kaiming_normal_(w_ho)

        self.readout.weight = torch.nn.Parameter(w_ho, requires_grad=True)


    def forward(self, x):

        batch_size = x.size(0)

        # Initialize hidden state
        hidden = self.init_hidden_state(batch_size)

        # Pass in input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)

        # Reshape outputs to be fit into the readout module
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.readout(out)

        # out = out.to(self.device)
        # hidden = hidden.to(self.device)

        return out, hidden


    def init_hidden_state(self, batch_size):
        # Initialize hidden state for the forward pass. This initialization
        # takes place at every batch.
        hidden = torch.zeros(self.n_layers,
                             batch_size,
                             self.hidden_size
                             )

        # hidden = hidden.to(self.device)

        return hidden.type(torch.float)


    def get_rnn_weights(self):
        return self.rnn.weight_hh_l0


    def set_rnn_weights(self, w):
        w = torch.tensor(w, dtype=torch.float)
        self.rnn.weight_hh_l0 = torch.nn.Parameter(w, requires_grad=True)

    # @property
    # def rnn_weights(self):
    #     print('getter method called')
    #     return self.rnn.weight_hh_l0.detach().numpy()
    #
    # def rnn_weights(self, w_hh):
    #     print('setter method called')
    #     self.rnn.weight_hh_l0 = torch.nn.Parameter(w_hh, requires_grad=True)


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
