import torch
import torch.nn as nn
from base import BaseModel
import gc

class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        # self.batch_size=batch_size
        self.padding = (filter_size - 1) // 2  # in this way the output has the same size
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input, hidden_state):
        hidden, c = hidden_state  # hidden and c are images with several channels
        # print 'hidden ',hidden.size()
        # print 'input ',input.size()
        combined = torch.cat((input, hidden), 1)  #concatenate in the channels
        # print 'combined',combined.size()
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)  # it should return 4 tensors
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1]).to(self.device),
                torch.zeros(batch_size, self.num_features, self.shape[0], self.shape[1]).to(self.device))


class CLSTM(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, shape, input_chans, filter_size, num_features, num_layers, batch_first=False):
        super(CLSTM, self).__init__()

        self.shape = shape  # H,W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.num_layers = num_layers
        cell_list = []
        cell_list.append(
            CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features).cuda()  if torch.cuda.is_available() else
            CLSTM_cell(self.shape, self.input_chans, self.filter_size, self.num_features))# the first
        # one has a different number of input channels

        for idcell in range(1, self.num_layers):
            cell_list.append(CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features).cuda() if torch.cuda.is_available() else
            CLSTM_cell(self.shape, self.num_features, self.filter_size, self.num_features))
        self.cell_list = nn.ModuleList(cell_list)
        self.batch_first = batch_first
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input, hidden_state):
        """
        args:
            hidden_state:list of tuples, one for every layer, each tuple should be hidden_layer_i,c_layer_i
            input is the tensor of shape seq_len,Batch,Chans,H,W
        """

        # now is seq_len,B,C,H,W
        current_input = input.transpose(0, 1) if self.batch_first else input
        # current_input=input
        next_hidden = []  # hidden states(h and c)
        seq_len = current_input.size(0)

        for idlayer in range(self.num_layers):  # loop for every layer

            hidden_c = hidden_state[idlayer]  # hidden and c are images with several channels
            all_output = []
            output_inner = []
            for t in range(seq_len):  # loop for every step
                hidden_c = self.cell_list[idlayer](current_input[t, ...],
                                                   hidden_c)  # cell_list is a list with different conv_lstms 1 for every layer

                output_inner.append(hidden_c[0])

            next_hidden.append(hidden_c)
            current_input = torch.cat(output_inner, 0).view(current_input.size(0),
                                                            *output_inner[0].size())  # seq_len,B,chans,H,W

        return next_hidden, current_input

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

class Net(BaseModel):
    def __init__(self, shape, input_chans, kernel_size, hidden_dim,
                 batch_size, n_past, n_fut, num_layer):
        super(Net, self).__init__()
        self.enc = Encoder(shape, input_chans, kernel_size, hidden_dim, num_layer, batch_size)
        self.dec = Decoder(shape, kernel_size, hidden_dim, num_layer)
        self.n_past = n_past
        self.n_fut = n_fut
        self.input_chans = input_chans
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    def forward(self, input_tensor):
        h = self.enc(input_tensor)
        x = torch.zeros((input_tensor.shape[0] , self.n_fut, self.hidden_dim, input_tensor.shape[3],input_tensor.shape[4])).to(self.device)
        out = self.dec(x, h)
        return out

class Encoder(nn.Module):
    def __init__(self, shape, input_chans, filter_size, hidden_dim, num_layers, batch_size):
        super().__init__()
        self.clstm = nn.ModuleList([CLSTM(shape, hidden_dim, filter_size, hidden_dim, 1, batch_first=True) for i in range(num_layers)])

        self.subnet = nn.ModuleList([nn.Conv2d(1, hidden_dim, 3, padding=1)])
        self.subnet.extend([nn.Conv2d(hidden_dim,hidden_dim,3,padding=1) for i in range(num_layers-1)])
        self.act = nn.LeakyReLU(0.2)
        self.num_layers = num_layers
        self.initial_hidden_state = self.clstm[0].init_hidden(batch_size)

    def bn(self, x, nc):
        batch_norm = nn.InstanceNorm3d(nc)
        x = x.permute(0,2,1,3,4)
        x = batch_norm(x)
        x = x.permute(0,2,1,3,4)
        return x
    
    def forward(self, input):
        h = self.initial_hidden_state
        hidden_states = []
        (b, sl) = input.shape[:2]
        for i in range(self.num_layers):
            input = torch.reshape(input, (-1, input.shape[2], input.shape[3], input.shape[4]))
            input = self.act(self.subnet[i](input))
            input = torch.reshape(input, (b, sl, input.shape[1], input.shape[2], input.shape[3]))
            input = self.bn(input, input.shape[2])
            state, input = self.clstm[i](input, h)
            hidden_states.append(state)
        return tuple(hidden_states)

class Decoder(nn.Module):
    def __init__(self, shape, filter_size, hidden_dim, num_layers):
        super().__init__()
        self.clstm = nn.ModuleList([CLSTM(shape, hidden_dim, filter_size, hidden_dim, 1, batch_first=True) for i in range(num_layers)])
        self.subnet = nn.ModuleList([nn.Conv2d(hidden_dim, hidden_dim,3,padding=1) for i in range(num_layers-1) ])
        self.subnet.append(nn.Conv2d(hidden_dim,1,1,padding=0))
        self.act = nn.LeakyReLU(0.2)
        self.sigm = nn.Sigmoid()
        self.num_layers = num_layers
    def bn(self, x, nc):
        batch_norm = nn.InstanceNorm3d(nc)
        x = x.permute(0,2,1,3,4)
        x = batch_norm(x)
        x = x.permute(0,2,1,3,4)
        return x
    
    def forward(self, input, h):
        (b, sl) = input.shape[:2]
        for i in range(self.num_layers):
            state, input = self.clstm[i](input, h[(i+1)*-1]) #todo check dimension
            input = input.transpose(0,1)
            input = torch.reshape(input, (-1, input.shape[2], input.shape[3], input.shape[4]))
            input = self.act(self.subnet[i](input)) if i < self.num_layers-1 else self.subnet[i](input)
            input = torch.reshape(input, (b, sl, input.shape[1], input.shape[2], input.shape[3]))
            if i < self.num_layers -1:
                input = self.bn(input, input.shape[2])
        return input
