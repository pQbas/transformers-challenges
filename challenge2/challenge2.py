from transformers_challenges import Challenge2, disp
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(- math.log(10000) * torch.arange(0, d_model, 2).float() / d_model) 
        pe = torch.zeros(max_len, d_model) 

        pe[:, 0::2] = torch.cos(position * div_term)
        pe[:, 1::2] = torch.sin(position * div_term)

        self.pos_encod = pe.unsqueeze(0)

    def forward(self, x):
        print(x.shape)
        print(self.pos_encod[:,:].shape)
        return x + self.pos_encod[:, :x.size()[1]]


d_model = 6
sequence_len = 4
token_embedding = torch.rand([sequence_len, d_model])

disp(token_embedding)
disp(torch.sin(torch.tensor([1])))
disp(torch.cos(torch.tensor([1])))

max_length = 12

disp(torch.arange(0, max_length))
print(torch.arange(0, max_length).shape)


disp(torch.arange(0, max_length).unsqueeze(1))
print(torch.arange(0, max_length).unsqueeze(1).shape)

disp(torch.exp(torch.tensor([math.log(1)])))
disp(torch.arange(0, d_model, 2).float())
disp(torch.arange(0, d_model, 2).float() / d_model)
disp(- math.log(10000) * torch.arange(0, d_model, 2).float() / d_model)


position = torch.arange(0, max_length).unsqueeze(1)
div_term = torch.exp(- math.log(10000) * torch.arange(0, d_model, 2).float() / d_model) 
even = torch.sin(position * div_term)

disp(even)

pe = torch.zeros(max_length, d_model)
disp(pe)

pe[:, 0::2] = torch.cos(position * div_term)
pe[:, 1::2] = torch.sin(position * div_term)

disp(pe)
disp(pe.unsqueeze(0))
# disp(token_embedding)
# disp(token_embedding.size()[0])
disp(token_embedding)
# disp(pe[:,:])

Challenge2.test(PositionalEncoding)

