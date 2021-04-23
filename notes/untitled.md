My notes trying to ensure that the code works properly.

- Adjacency Attention: tanh, relu
- Node Attention: tanh, relu 
- Feed Forward: relu, relu
- RESULT: flatline with train loss at 3.54, test loss at 3.26 at epoch 50+

- Adjacency Attention: tanh, relu
- Node Attention: tanh, relu 
- Feed Forward: relu, tanh  <<<<<< tanh instead of relu
- RESULT: test loss = 3.01 on best epoch.

- Adjacency Attention: tanh, tanh  <<<<<< tanh on 2nd instead of relu
- Node Attention: tanh, relu 
- Feed Forward: relu, tanh 
- RESULT: test loss = 3.06 on best epoch

- Adjacency Attention: tanh, relu
- Node Attention: relu, relu <<<<< change both to relu
- Feed Forward: relu, tanh
- RESULT: test loss flatlines with train loss at 3.54, test loss at 3.26 at epoc 50+

## REPEAT BEST 

- Adjacency Attention: tanh, relu
- Node Attention: tanh, relu 
- Feed Forward: relu, tanh 
- RESULT: test loss = 3.01 on best epoch #50.

------

## Remove wa matrix from attentive_adjacency

- Adjacency Attention: tanh, relu
- Node Attention: tanh, relu 
- Feed Forward: relu, tanh 
- RESULT: train loss at 3.54, test loss = 3.26 flatlined.

------

Reinstate wa matrix in attentive adjacency

- Adjacency Attention: tanh, relu
- Node Attention: tanh, relu 
- Feed Forward: relu, tanh 
- RESULT: test loss 3.01 at epoch 50

Average attention appears to be poorly correlate.

------

Double tanh on node attention

- Adjacency Attention: tanh, relu
- Node Attention: tanh, tanh   <<<<< changed here
- Feed Forward: relu, tanh 
- RESULT: test loss 3.02 at epoch 52



------

Attention neural net with relu, ident

_Taken from git log._

- Adjacency Attention: relu, ident 
- Node Attention: tanh, tanh 
- Feed Forward: relu, tanh 
- RESULT: test loss 3.05 at epoch 62

------

Remove 2nd hidden layer in feed forward

- Adjacency Attention: tanh, relu
- Node Attention: tanh, relu 
- Feed Forward: relu 
- RESULT: test loss = 2.94 on best epoch #50.

NEW BEST RECORD!

------

Relative to NEW BEST: remove wa parameter from adjacency attention

- Adjacency Attention: tanh, relu  <<<<< no wa parameter
- Node Attention: tanh, relu 
- Feed Forward: relu
- RESULT: test loss = 2.65 on best epoch #102.

NEW BEST RECORD!
