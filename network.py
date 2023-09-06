import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import socket

def is_network_available():
    try:
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        return False

   


device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set random seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# Define networks
class CNNRNNNet(nn.Module):

    def __init__(self, output_size, num_readouts = 2, hidden_size = 256):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_readouts = num_readouts

        # set up the CNN model
        if not is_network_available():
            pretrained_model_path_root = "/home/xuan/projects/def-bashivan/xuan/vision_model/pretrained_model/"
            pretrain_model = "resnet50-11ad3fa6.pth"
            self.cnnmodel = models.resnet50(pretrained=False).to(self.device) # cannot directly download on narval
            self.cnnmodel.load_state_dict(torch.load(pretrained_model_path_root + pretrain_model, map_location = torch.device(self.device)))
        else:
            self.cnnmodel = models.resnet50(pretrained=True)
        # freeze layers of cnn model
        for paras in self.cnnmodel.parameters():
            paras.requires_grad = False

        # get relu activation of last block of resnet50
        self.cnnmodel.layer4[2].relu.register_forward_hook(get_activation('relu'))

        self.cnnlayer = torch.nn.Conv2d(2048, hidden_size, 1) # we can also bring the resnet embedding dim to a number different from hidden size
        
        self.input_size = hidden_size*7*7
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.in2hidden = nn.Linear(self.input_size, hidden_size)
        self.layer_norm_in = nn.LayerNorm(self.hidden_size)
        
        self.rnn = nn.RNN(
            input_size = self.hidden_size, 
            hidden_size = self.hidden_size,
            )

        self.layer_norm_rnn = nn.LayerNorm(self.hidden_size)
        self.hidden2outputs = []
        for i in range(self.num_readouts):
            self.hidden2outputs.append(nn.Linear(self.hidden_size, self.output_size).to(self.device))
        # todo: do not consider noise here

    def forward(self, input_img, ):
        # preprocess image with resnet
        self.batch_size = input_img.shape[0]
        self.seq_len = input_img.shape[1]
     
        x = torch.swapaxes(input_img, 0, 1).float().to(self.device) # (seq_len, batchsize, nc, w, h)
        x_acts = []
        
        cnn_acts = []
        # print("brefore cnn internal activation")
        for i in range(self.seq_len):
            temp = self.cnnmodel(x[i,:,:,:,:])
            cnn_acts.append(activation["relu"])
            # print("actionvation shape:", activation["relu"].shape)
            x_act = self.cnnlayer(activation["relu"])
            # print("x act size:", x_act.shape)
            x_acts.append(x_act) # (batchsize, nc, w, h) = (batchsize, 2048, 7,7)
        
        x_acts = torch.stack(x_acts, axis = 0) # (seqlen, batchsize,nc, w,h)
        self.cnn_acts = torch.stack(cnn_acts, axis = 0) # (seqlen, batchsize, nc, w,h)
        self.cnn_acts_down = x_acts
        # print("after cnn internal activation")
        # x_acts_expand = torch.zeros((x_acts.shape[0], x_acts.shape[1], x_acts.shape[2], 8, 8))
        # x_acts_expand[:,:,:,:7,:7] = x_acts
        # x_acts = x_acts_expand
        # devide activations into patches and modify actions accordingly
        # k is the number of patches
        # print("value of self k:", self.k)
        # if self.k == 1:
        #     new_actions = actions
        # # print("shape of the activation:", x_acts.shape) # 
        # print("p3")       
        # new_x_acts = torch.zeros((int(x_acts.shape[0]*self.k), int(x_acts.shape[1]), x_acts.shape[2], int(self.patch_size), int(self.patch_size)))
        # # action size: batch size * seq len
        # new_actions = torch.ones((self.batch_size, int(self.seq_len * self.k)))*2
        # # print("what is self.k:", self.k)
        # sqrt_k = np.sqrt(self.k)
        # print("p4")
        # print("shape of the x act:", x_acts.shape)
        # for i in range(x_acts.shape[0]):
        #     for j in range(self.k):
        #         print("i,j:", i, j)
        #         w_idx = int(np.floor(j/sqrt_k))
        #         h_idx = int(j%sqrt_k)
        #         print(x_acts[i,:,:,w_idx*self.patch_size:(w_idx+1)*self.patch_size,h_idx*self.patch_size:(h_idx+1)*self.patch_size].shape)
        #         print(new_x_acts[i*self.k + j,:,:,:,:].shape)
        #         new_x_acts[i*self.k + j,:,:,:,:] = x_acts[i,:,:,w_idx*self.patch_size:(w_idx+1)*self.patch_size,h_idx*self.patch_size:(h_idx+1)*self.patch_size]
        #         if j == self.k - 1:
        #             new_actions[:,i*self.k + j] = actions[:,i]
        # # print("new actions shape:", new_actions.shape)
        # # print("new act shape:", new_x_acts.shape)
       
        
        x_acts = x_acts.reshape(x_acts.shape[0], self.batch_size, -1).to(self.device)
        # print("activation shape:", x_acts.shape)
        # print(task_index.unsqueeze(0).expand(x_acts.shape[0],-1, -1).shape)
       
       
       
        
        self.hidden_state = self.init_hidden(batch_size = self.batch_size).to(self.device) 
        # print("input to the networks size:", x_acts.shape)
        
        temp =  self.in2hidden(x_acts.float())
        # print("before RNN input layer")
        hidden_x = self.layer_norm_in(torch.relu(temp))
        # rnn_output, rnn_hn, = self.rnn(hidden_x, self.hidden_state )
        
        # print("before RNN layer")
        rnn_output, _ = self.rnn(hidden_x, self.hidden_state)
        # print("rnn_hn from the rnn layer:", rnn_hn.shape) # seq_len * batch size * hidden size
        # add noise to hidden layer
        # if is_noise:
        #     rnn_output = self.layer_norm_rnn(rnn_output + torch.randn(rnn_output.size()).to(self.device) * self.noise_std + self.noise_mean)
        # else:
        # print("after RNN layer")
        rnn_output = self.layer_norm_rnn(rnn_output)
        # print("out from the rnn layer:", rnn_output.shape)
        outs = []
        # todo: choice of where to put softmax activation
        for i in range(self.num_readouts):
            out = self.hidden2outputs[i](torch.tanh(rnn_output.to(self.device).to(self.device)))            
            outs.append(out)

        return torch.stack(outs)
        


    def init_hidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(1, batch_size, self.hidden_size))



