"""
Convolutional Long-Short Term Memeory Neural Network
Author          : SSI project team Wadden Sea
First Built     : 2021.08.01
Last Update     : 2021.11.05
Description     : This module is an implementation of Convolutional Long-Short
                  Term Memeory Neural Network (ConvLSTM).

                  The module is designed with reference to the script:
                  https://github.com/geek-yang/DLACs/blob/master/dlacs/ConvLSTM.py

                  More information can be found in the reference:
                  https://journals.ametsoc.org/view/journals/mwre/149/6/MWR-D-20-0113.1.xml
                  
Dependency      : os, numpy, pytorch
Return Values   : time series / array
Caveat!         : This module performs many-to-one prediction! It supports CUDA.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size,
                 weight_dict = None, cell_index = None):
        """
        Build convolutional cell for ConvLSTM.
        param input_channels: number of channels (variables) from input fields
        param hidden_channels: number of channels inside hidden layers, the dimension correponds to the output size
        param kernel_size: size of filter, if not a square then need to input a tuple (x,y)

        Parameters for creating ConvLSTM cells with given weights
        param weight_dict: weight matrix for the initialization of mu (mean)
        param cell_index: index of created BayesConvLSTM cell
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        
        # inherit weight matrix
        self.weight_dict = weight_dict
        self.cell_index = cell_index
        
        self.padding = int((kernel_size - 1) / 2) # make sure the output size remains the same as input
        # input shape of nn.Conv2d (input_channels,out_channels,kernel_size, stride, padding)
        # kernal_size and stride can be tuples, indicating non-square filter / uneven stride
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None
        
        # creating ConvLSTM cells with given weights
        if self.weight_dict is not None:
            # weight
            self.Wxi.weight.data = self.weight_dict['cell{}.Wxi.weight'.format(self.cell_index)].data
            self.Whi.weight.data = self.weight_dict['cell{}.Whi.weight'.format(self.cell_index)].data
            self.Wxf.weight.data = self.weight_dict['cell{}.Wxf.weight'.format(self.cell_index)].data
            self.Whf.weight.data = self.weight_dict['cell{}.Whf.weight'.format(self.cell_index)].data
            self.Wxc.weight.data = self.weight_dict['cell{}.Wxc.weight'.format(self.cell_index)].data
            self.Whc.weight.data = self.weight_dict['cell{}.Whc.weight'.format(self.cell_index)].data
            self.Wxo.weight.data = self.weight_dict['cell{}.Wxo.weight'.format(self.cell_index)].data
            self.Who.weight.data = self.weight_dict['cell{}.Who.weight'.format(self.cell_index)].data
            
            # bias
            self.Wxi.bias.data = self.weight_dict['cell{}.Wxi.bias'.format(self.cell_index)].data
            self.Wxf.bias.data = self.weight_dict['cell{}.Wxf.bias'.format(self.cell_index)].data
            self.Wxc.bias.data = self.weight_dict['cell{}.Wxc.bias'.format(self.cell_index)].data
            self.Wxo.bias.data = self.weight_dict['cell{}.Wxo.bias'.format(self.cell_index)].data            

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        #print(abs(self.Wci).sum(),abs(self.Wcf).sum(),abs(self.Wco).sum()) #Wci,Wcf,Wco are always 0!!!!
        return ch, cc #return the updated hidden=ch(t) and cell=cc(t) states from the input x(t) and states c(t-1) and h(t-1)

    #initialization of Wci,Wcf,Wco[1, hidden, height, width]=0 during the 1st epoch at t=0 for all the layers
    #initialization of the hidden and cell states h,c[batch_size, hidden, height, width] for all epochs at t=0 from a normal distribution (mean=0,std=1)
    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1])).to(device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.randn(batch_size, hidden, shape[0], shape[1])).to(device), 
                Variable(torch.randn(batch_size, hidden, shape[0], shape[1])).to(device))


class ConvLSTM(nn.Module):
    """
    This is the main ConvLSTM module.
    param input_channels: number of channels (variables) from input fields
    param hidden_channels: number of channels inside hidden layers, for multiple layers use tuple, the dimension correponds to the output size
    param kernel_size: size of filter, if not a square then need to input a tuple (x,y)
    param weight_dict: whether the model is initialized with given weights
    """
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, weight_dict = None):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels #=[input_channels,hidden_channels]
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if weight_dict is None: # initialize network without given weights
                cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            else:
                cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size,
                                   weight_dict, i)
            setattr(self, name, cell)
            self._all_layers.append(cell)        

    def forward(self, x, timestep, internal_state_hc=None):
        """
        Forward module of ConvLSTM.
        param x: input data with dimensions [batch size, channel, height, width]
        param timestep: parameter relates to the internal state initialization
                        if 0, then the internal state will be initialized!
        """
        if timestep == 0:
            self.internal_state = []
        # loop inside 
        for i in range(self.num_layers):
            # all cells are initialized in the first step
            name = 'cell{}'.format(i)
            if timestep == 0:
                if not internal_state_hc:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                else:
                    h = torch.tensor(internal_state_hc[i][0].detach().cpu().numpy()).to(device) 
                    c = torch.tensor(internal_state_hc[i][1].detach().cpu().numpy()).to(device)
                self.internal_state.append((h, c))
                         
            # do forward
            (h, c) = self.internal_state[i]
            x, new_c = getattr(self, name)(x, h, c)
            self.internal_state[i] = (x, new_c)
            #record as outputs: updated hidden state from last layer
            if i == (self.num_layers - 1):
                output = x 

        return output, self.internal_state.copy()
