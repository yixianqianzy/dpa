import torch
from torch.nn import functional as F
import torch.nn as nn

class Linear(nn.Linear): #used in MAML to forward input with fast weight 
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.fast = None #Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) #weight.fast (fast weight) is the temporaily adapted weight
        else:
            out = super(Linear, self).forward(x)
        return out

class selfmodel(torch.nn.Module):
    def __init__(self, SIDE_INFO_ITEM=2000, SIDE_INFO_USER=2000, emb_size=2000, fc_dim=1000):
        super(selfmodel, self).__init__()

        self.split = SIDE_INFO_ITEM
        self.side_info_y_item_emb = nn.Linear(SIDE_INFO_ITEM, emb_size)
        self.side_info_y_user_emb = nn.Linear(SIDE_INFO_USER, emb_size) 
        
        self.fc1 = Linear(emb_size*2, fc_dim)
        self.fc2 = Linear(fc_dim, fc_dim)
        self.linear_out = Linear(fc_dim, 1)
        
        self.final_part = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.linear_out, nn.Sigmoid())
    
    def forward(self, x):
        side_info_y_user, side_info_y_item = x[:, :self.split], x[:, self.split:]
        side_info_y_item_emb = F.relu(self.side_info_y_item_emb(side_info_y_item))
        side_info_y_user_emb = F.relu(self.side_info_y_user_emb(side_info_y_user))
        
        x = torch.cat((side_info_y_item_emb, side_info_y_user_emb), 1)
        x = self.final_part(x)
        return x
