import torch
from torch.autograd import Variable
from torch.nn import RNN, GRU, LSTM
import numpy as np
import math

from tqdm import tqdm

import time

def compute_LE(input_signal, rnn_model, k=100000, warmup=10, T_ons=None): #kappa=10, diff=10,

	# input signal - (batch/iter, seq_len/batch_size, features)
	x = Variable(input_signal, requires_grad=False).to(rnn_model.device)

	# initial hidden state
	n_iters, batch_size, input_size = x.shape
	h_0 = Variable(rnn_model.init_hidden_state(n_iters), requires_grad=False).to(rnn_model.device)

	# get number of layers and hidden size of rnn
	num_layers  = rnn_model.n_layers
	hidden_size = rnn_model.hidden_size

	# total number of Lyapunov exponents
	L = num_layers * hidden_size
	k = max(min(L, k), 1)

	# QR - algorithm
	Q = torch.reshape(torch.eye(L), (1, L, L)).repeat(n_iters, 1, 1).to(rnn_model.device)
	Q = Q[:, :, :k] #Choose how many exponents to track

	# initialize hidden state h_t
	h_t = h_0

    # initialization of r vals
	rvals = torch.ones(n_iters, batch_size, k).to(rnn_model.device) #storage

	# ------------------------
	# warmup
	t = 0
	if warmup !=0:
		for x_t in x.transpose(0,1)[:warmup]: # desc='\t\t':

			# # t-th element in the batch for all batches
	        # x_t = torch.unsqueeze(x_t, 1)

			# initialize network states
			states = (h_t,)

			# estimate Jacobian
			J = jac_RNN(rnn_model.rnn.all_weights,
						h_t, x_t,
						rnn_model.bias,
						)

			# update states
			_, states = oneStep(torch.unsqueeze(x_t, 1), *states, rnn_model)
			h_t = states

			# estimate Q matrix
			Q = torch.matmul(torch.transpose(J, 1, 2), Q)

	# apply QR decomposition to matrix Q
	Q, _ = torch.qr(Q, some=True)

	# ------------------------
	if T_ons is None: T_ons = 1

	#T_pred = math.log2(kappa/diff)
	#T_ons = max(1, math.floor(T_pred))

	# ------------------------
	# after warmup
	t_QR = t
	for x_t in x.transpose(0,1): #, desc='\t\t':

		if ((t - t_QR) >= T_ons) or (t == 0) or (t == batch_size):
			QR = True
		else:
			QR = False

		# # t-th element in the batch
		# x_t = torch.unsqueeze(x_t, 1)

		# initialize network states
		states = (h_t,)

        # estimate Jacobian
		J = jac_RNN(rnn_model.rnn.all_weights,
					h_t, x_t,
					rnn_model.bias,
					)

		# update states
		_, states = oneStep(torch.unsqueeze(x_t, 1), *states, rnn_model)
		h_t = states

		if QR:
			Q, r = oneStepVarQR(J, Q)
			t_QR = t
		else:
			Q = torch.matmul(torch.transpose(J, 1, 2), Q)
			r = torch.ones((n_iters, hidden_size))

		rvals[:, t, :] = r

		t += 1

	LE = torch.sum(torch.log2(rvals.detach()), dim=1)/batch_size

	return LE, rvals


def jac_RNN(rnn_weights, h, x, bias):

	device = get_device(h)

	if bias:
		W_i, W_h, b_i, b_h = regroup_weights(rnn_weights, bias)
		b = [b1 + b2 for (b1,b2) in zip(b_i, b_h)]
	else:
		W_i, W_h = regroup_weights(rnn_weights, bias)
		b = [torch.zeros(w.shape[0],).to(device) for w in W_i]

	num_layers, n_iters, hidden_size = h.shape
	input_size = x.shape[-1]

	# ------------------------
	h_in = h.transpose(1,2).detach()
	x_l = x.t()

	J = torch.zeros(n_iters, num_layers * hidden_size, num_layers * hidden_size).to(device)
	for layer in range(num_layers):

		y = (W_i[layer] @ x_l + W_h[layer] @ h_in[layer] + b[layer].repeat(n_iters,1).t()).t()
		x_l = torch.tanh(y).t()

		J_h = sech(y)**2 @ W_h[layer]
		J[:, layer*hidden_size:(layer+1)*hidden_size, layer*hidden_size:(layer+1)*hidden_size] = J_h

		if layer > 0:
			J_xt = sech(y)**2 @ W_i[layer]

			for l in range(layer, 0, -1):
				J[:, layer*hidden_size:(layer+1)*hidden_size, (l-1)*hidden_size:l*hidden_size] = J_xt @ J[:, (layer-1)*hidden_size:(layer)*hidden_size, (l-1)*hidden_size:l*hidden_size]

	return J


def regroup_weights(weights, bias):
	# weigthts should be a list containing the weights for each layer.
	# each layer should be organized as (W_i, W_h, b_i, b_h).
	num_layers 	= len(weights)

	W_i = []
	W_h = []
	b_i = []
	b_h = []

	if bias:
		param_list = (W_i, W_h, b_i, b_h)
	else:
		param_list = (W_i, W_h)

	grouped = []
	for idx, param in enumerate(param_list):
		for layer in range(num_layers):
			param.append(weights[layer][idx].detach())
		grouped.append(param)

	return grouped


def sech(X):
	device = get_device(X)
	return torch.diag_embed(1/(torch.cosh(X)))


def oneStep(x, h, model):
	return model(x, h)


def oneStepVarQR(J, Q):
	Z = torch.matmul(torch.transpose(J, 1, 2), Q) #Linear extrapolation of the network in many directions
	q, r = torch.qr(Z, some=True) #QR decomposition of new directions
	s = torch.diag_embed(torch.sign(torch.diagonal(r, dim1=1, dim2=2)))#extract sign of each leading r value

	return torch.matmul(q, s), torch.diagonal(torch.matmul(s, r), dim1 = 1, dim2 = 2) #return positive r values and corresponding vectors


def get_device(X):
    if X.is_cuda:
        return torch.device('cuda')
    else:
        return torch.device('cpu')
