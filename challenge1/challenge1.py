from transformers_challenges import Challenge1, disp
import torch
import torch.nn as nn

vocab_size = 2
d_model = 3
matrix = torch.rand(vocab_size, d_model)

disp(matrix)
disp(matrix[1])
disp(matrix[1,2])
disp(matrix[[1,0,1]])
disp(matrix[[[1,0,1],[1,0,0]]])

input_tokens = torch.tensor([[1,0,1], [1,0,0]])

disp(input_tokens)
disp(matrix[input_tokens])


class EmbeddingLayer(nn.Module):
    def __init__ (self, d_model, vocab_size) :
        super().__init__()
        self.embedding_matrix = torch.rand(vocab_size, d_model, requires_grad = True)

    def forward(self, x):
        return self.embedding_matrix[x]


Challenge1.test(EmbeddingLayer)

