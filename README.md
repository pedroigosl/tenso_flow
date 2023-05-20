# tenso_flow

Practice MLP project trying to mimic tensorflow modularity

## +++ DISCLAIMER: +++

<span style="color:red">**THIS CODE IS NOT IN ANY WAY, SHAPE OR FORM RELATED TO TENSORFLOW, GOOGLE OR ALPHABET**</span>. There is not a single line of their code in it, and it's been only inspired by tensorflow modularity and tool naming scheme as a gag and challenge to myself. Tenso_flow is a pun on tensorflow, where **tenso** stands for **tense**, as in stressed ~~very much how i felt during the entire project~~, in portuguese, my mother tongue.

Please don't sue me poor 3rd world boi ;-;

This library was made with the objective of practicing programming and studying machine learning. It is NOT supposed TO BE USED and as a tool, is mostly useless. It is inefficient, lacks most basic functionalities and there is almost no protection against user error. 

All that said, this was made with much care and i tried to make it as modular and generic as possible (for my base cases).

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

## Acknowledgements:

Results are reproducible by repeating its seed. Seeds from sample code were cherrypicked to reduce training time, since it is not very efficient and i am impatient.

On sample file, both training and testing are done on full population, as i didn't implement train/test split inside the module and the focus is just to show it converging, not a real world application. That said, on the first 2 datasets, decision surfaces are observable in plots. 

## Tutorial:

For **how to use**, refer to tests notebook sample.

Why do you want to learn how to use it anyway? Again, it isn't useful and just for showcase
