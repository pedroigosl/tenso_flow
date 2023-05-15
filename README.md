# tenso_flow
Practice MLP project trying to mimic tensorflow modularity

## +++ DISCLAIMER: +++
<span style="color:red">**THIS CODE IS NOT IN ANY WAY, SHAPE OR FORM RELATED TO TENSORFLOW, GOOGLE OR ALPHABET**</span>. There is not a single line of their code in it, and it's been made based only in tensorflow modularity and tool naming scheme as a gag and challenge to myself. Tenso_flow is a pun on tensorflow, where **tenso** stands for **tense**, as in stressed ~~very much how i felt during the entire project~~, in portuguese, my mother tongue.

Please don't sue me ;-;

This library was made with the objective of practicing programming and studying machine learning. It is NOT supposed TO BE USED and as a tool, is mostly useless. It is inefficient, lacks most basic functionalities and there is almost no protection against user error. 

All that said, this was made with much care and i tried to make it as modular and generic as possible.

## Functionalities:

As long as your dataset is:

- in a pd.DataFrame format, 
- all fields numerical and 
- has the right formats for input and output

It must be able to complete simple classification and regression tasks, with any hidden layers and topology desired 

At this point in time (and probably forever), there is:

- Layer types: Input, Dense
- Activation functions: Linear, Step, Sigmoid, Tanh
- Aggregation functions: Sum, Mean
- Optimizers: StochasticGradientDescent
- Loss Functions: MeanSquaredError, AbsoluteSquaredError
- Models: Sequential

Results are reproducible by repeating its seed

For **how to use**, refer to xor problem notebook sample
