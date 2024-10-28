
# Challenge 3 : Implement Scaled Dot-Product Attention

## Description

In Transformer model, attention is a mechanism that allows each token in a sequence to
focus on aother tokens, capturing relationships and dependencies. The `scaled dot-product`
function is central tot his mechanism, computing attention weights based on a set of
`queries (Q)`, `keys (K)` and `values (V)`.

- **Query (Q)** : Represents the token (or word) for which we want to compute attention.
- **Key (K)** : Represents the token that could receive attention from the query.
- **Value (V)** : Represents the content or information associated with each key.

The scaled dot-product function calculates how much attention the `query` should 
pay to each `key`. using a dot product similarity measure sacled by the square root
of the dimensionality. This scalling helps to keeps values within a manageable range,
improving gradient flow and training stability.

## Mathematical Formulation

Given:
-  $\LARGE Q$ : A tensor representing queries of shape $\Large \( N, T, d_k \)$
-  $\LARGE K$ : A tensor representing keys of shape $\Large \(N, T, d_k \)$
-  $\LARGE V$ : A tensor representing values of shape $ \Large \(N, T, d_k \)$

where:

-  $\LARGE N$ is the batch size
-  $\LARGE T$ is the sequence length
-  $\LARGE d_k$ is the dimensionality of the queries and keys
-  $\LARGE d_v$ is the dimensionality of the values

The steps are as follows

1. Compute attention scores:

    - The attention score is computed by taking the doc product between each query
      vector $\Large Q_i$ and each key vector $\Large K_j$ for a given sqequence.

    - The dot product is scaled by the factor $\Large \sqrt{d_k}$ to  prevent extremely large
      values and stabilize the gradients.

    - Here `Q` and `K` are matrix-multiplied, within $K^T$ being the transfosed of `K`.

$$
\LARGE \text{scores} = \frac{Q \cdot K^T}{\sqrt{d_k}}
$$

2. Apply Softmax to obtain attention weights:

    - Apply the softmax function along the last dimension to convert scores into
    probabilities. This step normalizes the scores so that they sum to 1, making it
    easier to interpret each as an "attention weight".

$$
\LARGE \text{Attention Weights} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right)  
$$

3. Multiply attention weights by Values (V):
    - Multiply the attention weights by the value matrix $V$ to focus on sepcific parts of $V$
    based on the attention weights. 

$$
\LARGE  \text{Output} = \text{Attention Weights} \cdot V 
$$

This produces the output of the scaled dot-product attention, which has the shape $(N, T, d_v)$
where each element represents a combination of the values weighted by the attention each query
paid to each key.

## Instructions

1. Define a function `scaled_dot_product_attention(Q. K. V)`
2. Calculate the attention scores by taking the dot product of `Q` and the transponse of `K`.
3. Scale the scores by `1 / sqrt(d_k)`
4. Apply `softmax` to the scaled scores to get attention wieghts
5. Multiply the attention weights by `V` to get the outpu.
6. Return both, the output and the attention weights




