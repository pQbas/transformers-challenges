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
- $\Large d_{model}$: embedding dimmension size
- $\Large \text{pos}$: position of a token in the sequence, in the range $[0, \text{max len}]$
- $\Large i$: dimension index in the embedding vector, in range $[0, d_{model}]$


We define positional encodeing $PE(pos,i)$ as follows:

$$
\Huge
PE (pos, i) = 
\begin{cases} 
\sin\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right) & \text{if } i \text{ is even} \\
\cos\left(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}\right) & \text{if } i \text{ is odd} 
\end{cases}
$$

This results in a matrix $\Large PE$ of shape `(max_len, d_model)`, where each row corresponds
to the positional encoding for specific position.

## Instructions:
 1. Define a `PositionalEncoding` class with parameters `d_model, max_len`.
 2. In the constructor, compute the positional encoding $PE$ for a maximum sequence lenght, `max_len`,
    using formulas provided.
 2. In the forward method:
    - Add the positional encoding $PE$ to the input embeddings
    - Ensure the output has the same shape as the input embeddings 
      `(batch_size, sequence_length, d_model)`.  

