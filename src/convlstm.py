"""
Convolutional Long-Short Term Memeory Neural Network
Author          : SSI project team Wadden Sea
First Built     : 2021.08.01
Last Update     : 2021.11.27
Description     : This module is an implementation of Convolutional Long-Short
                  Term Memeory Neural Network (ConvLSTM).

                  The module is designed with reference to the script:
                  https://github.com/geek-yang/DLACs/blob/master/dlacs/ConvLSTM.py

                  More information can be found in the reference:
                  https://journals.ametsoc.org/view/journals/mwre/149/6/MWR-D-20-0113.1.xml
                  
ML Library      : pytorch
Return Values   : pytorch tensor with t,y,x dependence
Caveat!         : This module performs many-to-one prediction! It supports CUDA.
"""

import torch
import torch.nn as nn

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size,
                 weight_dict = None, cell_index = None, num_time_series=0):
        """
        Build convolutional cell for ConvLSTM.
        Inputs:
        - input_channels: number of channels (variables) from input fields
        - hidden_channels: number of channels inside hidden layers, the dimension correponds to the output size
        - kernel_size: size of filter, if not a square then need to input a tuple (x,y)
        - num_time_series: number of inputs that are only time series (without x,y dependence)

        Parameters for creating ConvLSTM cells with given weights:
        - weight_dict: weight matrix for the initialization of mu (mean)
        - cell_index: index of created ConvLSTM cell (also used when num_time_series>0)
        """
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_time_series = num_time_series
        
        # inherit weight matrix
        self.weight_dict = weight_dict
        self.cell_index = cell_index
        
        self.padding = int((kernel_size - 1) / 2) # make sure the output size remains the same as input
        # input shape of nn.Conv2d (input_channels,out_channels,kernel_size, stride, padding)
        # kernal_size and stride can be tuples, indicating non-square filter / uneven stride

        #when some inputs are only time series (don't have spatial structure) use kernle_size=1, 
        #it will be equivalent to a nn.Linear layer.
        if self.num_time_series>0 and self.cell_index==0:
            self.Wxi = nn.Conv2d(self.input_channels-self.num_time_series, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
            self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
            self.Wxf = nn.Conv2d(self.input_channels-self.num_time_series, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
            self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
            self.Wxc = nn.Conv2d(self.input_channels-self.num_time_series, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
            self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
            self.Wxo = nn.Conv2d(self.input_channels-self.num_time_series, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
            self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        else:
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

        #when some inputs are only time series (don't have spatial structure) use kernle_size=1, 
        #it will be equivalent to a nn.Linear layer.
        if self.num_time_series>0 and self.cell_index==0:
            self.Wxit = nn.Conv2d(self.num_time_series, self.hidden_channels, 1, 1, 0, bias=False)
            self.Wxft = nn.Conv2d(self.num_time_series, self.hidden_channels, 1, 1, 0, bias=False)
            self.Wxct = nn.Conv2d(self.num_time_series, self.hidden_channels, 1, 1, 0, bias=False)
            self.Wxot = nn.Conv2d(self.num_time_series, self.hidden_channels, 1, 1, 0, bias=False)
        
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
            # weight for inputs that are only time series
            if self.num_time_series>0 and self.cell_index==0:
                self.Wxit.weight.data = self.weight_dict['cell{}.Wxit.weight'.format(self.cell_index)].data     
                self.Wxft.weight.data = self.weight_dict['cell{}.Wxft.weight'.format(self.cell_index)].data   
                self.Wxct.weight.data = self.weight_dict['cell{}.Wxct.weight'.format(self.cell_index)].data     
                self.Wxot.weight.data = self.weight_dict['cell{}.Wxot.weight'.format(self.cell_index)].data   

    def forward(self, x, h, c):
        if self.num_time_series>0 and self.cell_index==0:
            ci = torch.sigmoid(self.Wxi(x[:1,:-self.num_time_series,...]) + self.Wxit(x[:1,-self.num_time_series:,...]) + self.Whi(h) + c * self.Wci)
            cf = torch.sigmoid(self.Wxf(x[:1,:-self.num_time_series,...]) + self.Wxft(x[:1,-self.num_time_series:,...]) + self.Whf(h) + c * self.Wcf)
            cc = cf * c + ci * torch.tanh(self.Wxc(x[:1,:-self.num_time_series,...]) + self.Wxct(x[:1,-self.num_time_series:,...]) + self.Whc(h))
            co = torch.sigmoid(self.Wxo(x[:1,:-self.num_time_series,...]) + self.Wxot(x[:1,-self.num_time_series:,...]) + self.Who(h) + cc * self.Wco)
            #print("calling new module", self.cell_index, self.num_time_series)
        else:
            ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
            cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
            cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
            co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        #print(abs(self.Wci).sum(),abs(self.Wcf).sum(),abs(self.Wco).sum()) #Wci,Wcf,Wco are always 0!!!!
        #return the updated hidden=ch(t) and cell=cc(t):
        #-for the 1st layer: from the input x(t) and states c(t-1) and h(t-1)
        #-for the nth layer: from input x(t)=output h(t) of the previous layer and states c(t-1) and h(t-1) of this layer
        return ch, cc 

    #initialization of Wci,Wcf,Wco[1, hidden, height, width]=0 during the 1st epoch at t=0 for all the layers
    #initialization of the hidden and cell states h,c[batch_size, hidden, height, width] for all epochs at t=0 from a normal distribution (mean=0,std=1)
    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = torch.zeros(1, hidden, shape[0], shape[1], device=device) #create a tensor directly on the specified device
            self.Wcf = torch.zeros(1, hidden, shape[0], shape[1], device=device)
            self.Wco = torch.zeros(1, hidden, shape[0], shape[1], device=device)
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (torch.randn(batch_size, hidden, shape[0], shape[1], device=device), 
                torch.randn(batch_size, hidden, shape[0], shape[1], device=device))


class ConvLSTM(nn.Module):
    """
    This is the main ConvLSTM module.
    Inputs:
    - input_channels: number of channels (variables) from input fields
    - hidden_channels: number of channels inside hidden layers, the dimension correponds to the output size
    - kernel_size: size of filter, if not a square then need to input a tuple (x,y)
    - num_time_series: number of inputs that are only time series (without x,y dependence)

    Parameters for creating ConvLSTM cells with given weights:
    - weight_dict: weight matrix for the initialization of mu (mean)
    """
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, weight_dict = None, num_time_series=0):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels #=[input_channels,hidden_channels]
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.num_time_series = num_time_series
        self._all_layers = []

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            if weight_dict is None: # initialize network without given weights
                if self.num_time_series>0:
                    cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, None, i, self.num_time_series)
                else:
                    cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, None, None, self.num_time_series)               
            else:
                cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size, weight_dict, i, self.num_time_series)
            setattr(self, name, cell)
            self._all_layers.append(cell)        

    def forward(self, x, timestep, init_internal_states_hc=None):
        """
        Forward module of ConvLSTM.
        Inputs:
         - x = Input data with dimensions [batch size, input channel, height, width].
                 * batch_size = 1: is used to have continuous states when passing data to the model. 
                 * batch_size > 1: is used when the subset of input data (with  size = batch_size) 
                                   given to the model are consider as independent samples, so they 
                                   don't share contiguous internal states. 
                                   E.g.: 
                                   - Numerical simulations with slightly different initial 
                                     conditions (realizations of the flow).
                                   - Prediction of particle trajectories, each trajectory is an 
                                     independet sample (but only with LSTM).
                                   - Translation of sentences (but using only LSTM).
         - timestep = Parameter related to the initialization of the model and the internal states.
         - init_internal_states_hc = Give to the model initial hidden and cell states (default = None),
                                     useful for having continuous states after a backward propagation.
           If timestep = 0 and init_internal_states_hc = None: 
              * The model is initialized.
              * The internal hidden h and cell c states are initialized from a normal distribution.
           If timestep = 0 and init_internal_states_hc = list of [h0,c0] tensors: 
              * The model is initialized.
              * The internal hidden h and cell c states are given from h0 and c0.
           If timestep > 0: update h and c using the ConvLSTMCell.forward
        Outputs:
         - output = updated hidden state from last layer  [batch size, output channel, height, width].
         - internal_states = list of updated h and c for all hidden layers.
        Comments:
         - There is no sequence parameter, so you should build an external function to mimic this behaviour.
           The amount of times you call the function without reseting the internal states and before a backward
           propagation will be the sequence length; it can be incresed by using continuous states after a back prop.
        """
        if timestep == 0: 
            self.internal_states = [] #this reset the model in case init_internal_states_hc is given 
        # loop for all the layers
        for i in range(self.num_layers):
            # all cells are initialized in the first timestep
            name = 'cell{}'.format(i)
            if timestep == 0:
                if init_internal_states_hc:
                    h, c = init_internal_states_hc[i][0].detach(), init_internal_states_hc[i][1].detach()
                else:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                self.internal_states.append((h, c))
                         
            # do forward
            (h, c) = self.internal_states[i]
            x, new_c = getattr(self, name)(x, h, c)
            self.internal_states[i] = (x, new_c)
            #record as output: updated hidden state from last layer
            if i == (self.num_layers - 1):
                output = x 

        return output, self.internal_states.copy()
