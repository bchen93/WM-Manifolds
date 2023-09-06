import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from task import WM_task
from network import CNNRNNNet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--root', '-r', type=str, required=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_folder = "/home/xuan/projects/def-bashivan/xuan/AC2023/results/"
batch_size = 16
seq_len = 2
grid_size = 2
order_type = 1

# create the task datset
train_dataset = WM_task(image_path=args.root,seq_len = 2, delay_period = 1, grid_size = grid_size, mode = "train", order_type = order_type)
val_dataset = WM_task(image_path=args.root,seq_len = 2, delay_period = 1, grid_size = grid_size, mode = "val", order_type = order_type)
test_dataset = WM_task(image_path=args.root,seq_len = 2, delay_period = 1, grid_size = grid_size, mode = "test", order_type = order_type)

loaders = {"train": DataLoader(train_dataset, batch_size= batch_size, shuffle = True),
           "test": DataLoader(test_dataset, batch_size= 1, shuffle = False),
           "val": DataLoader(val_dataset, batch_size= batch_size, shuffle = False) # todo: change it back
           }

# define the network
network = CNNRNNNet(output_size = grid_size ** 2 + 1, num_readouts = seq_len, hidden_size = 256).to(device)
optimizer = optim.AdamW(network.parameters(), lr = 1e-3 )
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [50,100,150,400,500,], gamma = 0.5)

criterion = torch.nn.CrossEntropyLoss()

train_loss = []
val_loss = []
train_acc = []
val_acc = []

for epoch in range(10):
    print("epoch")
    for mode in ["train", "val"]:
        
        for i, input_collc in enumerate(loaders[mode]): 
            print(i)
            frames, actions = input_collc # actions: batch_size * seq_len * n_choices
            outputs = network(torch.tensor(frames).to(device)) # readout_heads * seq_len * batchsize * outptusize 
            outputs = outputs.permute(2,0,1,3)
            
            accs = []
            if order_type == 1:
                for ti in range(seq_len):
                    if ti == 0:
                        loss = criterion(outputs[:,ti,:,:].reshape(-1,outputs[ti].shape[-1]).cpu(), actions[:,ti,:].reshape(-1).to(torch.long))
                    else:
                        loss += criterion(outputs[:,ti,:,:].reshape(-1,outputs[ti].shape[-1]).cpu(), actions[:,ti,:].reshape(-1).to(torch.long))
                    predicted_action = torch.argmax(outputs[:,ti,:,:].reshape(-1,outputs[ti].shape[-1]).cpu(), axis = -1)
                    
                    accs.append(torch.sum(predicted_action == actions[:,ti,:].reshape(-1))/len(predicted_action))
            elif order_type == 0: # random order case; to be implemented
                for ti in range(seq_len):
                    predicted_action = torch.argmax(outputs[:,ti,:,:].reshape(-1,outputs[ti].shape[-1]).cpu(), axis = -1)
                    for curr_action in actions: # all permutations of actions
                        pass # not sure if there is an easy way to by pass the for loop

            if mode == "train":
                train_loss.append(loss.detach().cpu())
                train_acc.append(np.mean(accs))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                val_loss.append(loss.detach().cpu())
                val_acc.append(np.mean(accs))
 
        
        if mode == "train":
            scheduler.step()
    
    # save the model, loss and acc
    if val_acc[-1] > 0.99:
        print("save the model...")
        savename = os.path.join(save_folder, "ordered_checkpoint.pth")
        save_dict = {'epoch':epoch,
                "model_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": train_loss,
                "train_acc": train_acc,
                
                "val_loss": val_loss,
                "val_acc": val_acc,
                
                }
        torch.save(save_dict, savename)
                   


# collecting activation data
mode = "test"
activations = {}
def save_activation(name):
        def hook(model, input, output):
            activations.append(output[0].detach().cpu().numpy())

        return hook

network.rnn.register_forward_hook(save_activation('rnn'))

saved_data = {
        "predicted_action": [],
        "corrected_action": [],
        "activation": [],
    }


for i, input_collc in enumerate(loaders[mode]): 
    frames, actions = input_collc # actions: batch_size * seq_len * n_choices
    outputs = network(torch.tensor(frames).to(device)) # readout_heads * seq_len * batchsize * outptusize 
    outputs = outputs.permute(2,0,1,3)

    print("which activation to save:", activations[-1].shape)
    saved_data["activation"].append(activations[-1][:, 0, :])
    

    predicted_actions = []
    actions = []
    for ti in range(seq_len):
        actions.append(actions[:,ti,:].reshape(-1))
        predicted_actions.append(torch.argmax(outputs[:,ti,:,:].reshape(-1,outputs[ti].shape[-1]).cpu(), axis = -1))
    saved_data["predicted_action"].append(torch.stack(predicted_actions))
    saved_data["corrected_action"].append(torch.stack(actions))

# convert dict to df and save the file
df = pd.DataFrame(saved_data)

# Save the dataframe to a file (e.g., CSV format)
output_file = os.path.join(save_folder, "activations.pkl")
df.to_pickle(output_file)