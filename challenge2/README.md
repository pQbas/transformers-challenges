# Challenge 2: Implement Positional Encoding

## Description:

In a Transformer model, positional information is not inherently captured since it 
processes tokens in parallel rather than sequentially. To provide each token’s position in 
the sequence, we add a positional encoding to each token embedding. This encoding helps the 
model distinguish tokens based on their positions, allowing it to understand word order and 
structure.

Positional encoding is based on sinusoidal functions and can be calculated using the 
following mathematical expressions. The sinusoidal nature of this encoding allows the model 
to generalize to sequences longer than those seen during training.


Given:

- $pos$: position of a token in the sequence
- $i$: dimension index in the embedding vector
- $d_model$: embedding dimmension size

We define positional encodeing $PE_{pos,i}$ as follows:

$$
PE_{\text{pos}, i} = 
\begin{cases} 
\sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right) & \text{if } i \text{ is even} \\
\cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right) & \text{if } i \text{ is odd} 
\end{cases}
$$

This results in a matrix $PE$ of shape `(max_len, d_model)`, where each row corresponds
to the positional encoding for specific position.

## Instructions:
 1. Define a `PositionalEncoding` class.
 2. In the constructor, compute the positional encoding for a maximum sequence lenght, `max_len`,
    using formulas provided.
 2. In the forward method:
    - Add the positional encoding to the input embeddings
    - Ensure the output has the same shape as the input embeddings 
      `(batch_size, sequence_length, d_model)`.  

