# Challenge 1: Implement a Token Embedding Layer

## Description:

In natural language processing, words or tokens are represented as indices in a vocabulary. 
However, neural networks work better with dense vector representations, so each token index 
needs to be converted into a dense vector called an embedding. This embedding has a specific 
dimension `d_model` and allows the model to capture semantic information.

## Goal:

You will create a class EmbeddingLayer that:

1. Takes a vocabulary size $vocab_size$ and an embedding dimension $d_{model}$ as inputs.
2. Converts each token index to a dense vector (embedding) of shape $d_{model}$ 


## Instructions:

1. Define a class EmbeddingLayer. In the constructor, initialize the embedding matrix $E$ 
using random values. To initialize a matrix of shape (vocab_size, d_model), you can use 
torch.rand.

2. Define a forward method that accepts token indices as input. This method should:
    - Retrieve the embeddings for the token indices using the embedding matrix.
    - Return the dense embeddings with shape (batch_size, sequence_length, d_model).
