import time
import copy
from decimal import *
from datetime import timedelta
from typing import List, Callable

import numpy as np
import pandas as pd

def _equalsign_percent(bins = 30, 
                       value = 0, 
                       total = 100):
    
    equal_count = int((value/total)*bins)
    
    return f"[{equal_count*'='}{(bins - equal_count)*'.'}]"

# TODO: Add more time units (min and hour)
def _perftime_str(delta: int):
    
    delta = timedelta(seconds=delta)
    if (delta.seconds):
        seconds = delta.seconds
        microseconds = delta.microseconds
        return f"{seconds}s {microseconds}ms"
    if (delta.microseconds):
        microseconds = delta.microseconds
        return f"{microseconds}ms"

    
class Activation():
        
    class step():
        
        def __call__(self, 
                     input: float):
            
            if input >= 0: return 1
            return 0
        
        def gradient(self, 
                     input: float):
            
            return 1
    
    class linear():
        
        def __call__(self, 
                     input: float):
            
            return input
        
        def gradient(self, 
                     input: float):
            
            return 1
            
    class sigmoid():
        
        def __call__(self, 
                     input: float):
            
            try:
                x = Decimal(input)
                return float(1/(1 + Decimal.exp(-x)))

            except:
                print("Overflow: ", input)
                raise

        def gradient(self, 
                     input: float):
            
            x = input
            return x*(1-x)

    class tanh():
        
        def __call__(self, 
                     input: float):
            
            try:
                x = Decimal(input)
                return float((Decimal.exp(x) - Decimal.exp(-x))/(Decimal.exp(x) + Decimal.exp(-x)))

            except:
                print("Overflow: ", input)
                raise

        def gradient(self, 
                     input: float):
            
            x = input
            return 1-x**2
        
class Aggregation():
    
    class sum():
        
        def __call__(self, 
                     input: List[float]):
            
            return np.sum(input)
    class mean():
        
        def __call__(self, 
                     input: List[float]):
            
            return np.mean(input)
        

class Neuron():
    
    def __init__(self, 
                 input_size: int, 
                 activation: Activation,
                 aggregation: Aggregation,
                 bias: float = -1.0,):
        
        self._input_size = input_size
        self.random_weights()
        self.activation_func = activation
        self.aggregation_func = aggregation
        self.bias = bias   
           
    def set_weights(self, 
                    weights: List[float]):
        
        self._weights = weights
        
    def get_weights(self):
        
        return self._weights
    
    def get_output(self, 
                   inputs: List[float]):
        
        weights_vector = [self._weights[i]*input for i, input in enumerate(inputs)]
        weights_vector.append(self._weights[-1]*self.bias)
        aggregated = self.aggregation_func(weights_vector)
        output = self.activation_func(aggregated)
        return output  
      
    def random_weights(self):
        self._weights = [np.random.randint(0,10)/10 for _ in range(self._input_size + 1)]
        
class Layers():
    
    class Input():
        """ Placeholder input layer
        
        Empty placeholder layer for feature input shape definiton on topology
        """      
          
        def __init__(self, 
                     input_shape: int):
            
            self.input_shape = input_shape
            self.type = "input"
    
    class Dense():
        
        def __init__(self, 
                     units: int, 
                     activation: Activation, 
                     aggregation: Aggregation = Aggregation.sum(),
                     input_shape: int = None, 
                     use_bias: bool = True):
            """ Instantiates layer object with desired configurations

            Args:
                units (int): Neuron units count
                activation (Activation): Neurons' activation function
                aggregation (Aggregation, optional): Neurons' aggregation function. Defaults to Aggregation.sum().
                input_shape (int, optional): layer input shape. Defaults to None.
                use_bias (bool, optional): Whether to use bias. Defaults to True.
            """            
            
            self.activation = activation
            self.aggregation = aggregation
            self.unit_count = units
            self.output_shape = units
            self.input_shape = input_shape
            self.use_bias = use_bias
            self.type = "dense"
        
        def initialize(self, 
                       input_shape: int):
            """ Initializes layer neurons
            
            Initializes layer neurons using configurations declared during instantiation

            Args:
                input_shape (int): layer input shape to which neurons must be dimensioned

            Returns:
                List[Neuron]: Layer of instantiated neurons
            """            
            
            if not self.use_bias:
                units = [Neuron(input_size = input_shape, activation = copy.deepcopy(self.activation), aggregation=copy.deepcopy(self.aggregation), bias = 0) for _ in range(self.unit_count)] 
            else: 
                units = [Neuron(input_size = input_shape, activation = copy.deepcopy(self.activation), aggregation=copy.deepcopy(self.aggregation)) for _ in range(self.unit_count)]
            return units

class Error():
    
    class Difference():
        
        def __init__(self):
            ...
        
        def __new__(cls, 
                    expected: List[float], 
                    predicted: List[float]):
            
            return [(exp - pred) for exp, pred in zip(expected, predicted)]
    class Squared():
        
        def __init__(self):
            ...
        
        def __new__(cls, 
                    expected: List[float], 
                    predicted: List[float]):
            
            return [(((exp - pred)**2)/2) for exp, pred in zip(expected, predicted)]
        
class Losses():
    
    class MeanSquaredError():
        
        def __init__(self):
            ...
        
        def get_loss(dataset: pd.DataFrame, 
                     net: List[List[Neuron]], 
                     output_shape: int, 
                     run_func: Callable = None):
            
            loss_vector = []
            for i in range(len(dataset)):
                row = list(dataset.iloc[i,:])
                if not run_func:
                    predicted = net.get_output(row[:-output_shape])
                else:
                    predicted = run_func(row[:-output_shape])
                expected = row[-output_shape:]
                loss_vector.append([expected, predicted])
            return np.mean([sum([(exp - pred)**2 for exp, pred in zip(expected, predicted)]) for expected, predicted in loss_vector])
        
    class MSE(MeanSquaredError):
        ...

    class MeanAbsoluteError():
        
        def __init__(self):
            ...
        
        def get_loss(dataset: pd.DataFrame, 
                     net: List[List[Neuron]], 
                     output_shape, 
                     run_func = None):
            
            loss_vector = []
            for i in range(len(dataset)):
                row = list(dataset.iloc[i,:])
                    
                if not run_func:
                    predicted = net.get_output(row[:-output_shape])
                else:
                    predicted = run_func(row[:-output_shape])
                expected = row[-output_shape:]

                loss_vector.append([expected, predicted])
            return np.mean([sum([(exp - pred) for exp, pred in zip(expected, predicted)]) for expected, predicted in loss_vector])
        
    class MAE(MeanAbsoluteError):
        ...
        
    class Placeholder():
        def __init__(self):
            ...
        
        def get_loss(dataset: pd.DataFrame, 
                     net: List[List[Neuron]], 
                     output_shape: int, 
                     run_func: Callable = None):
            return 0

class Optimizers():
    
    class GradientDescent():
        
        def __init__(self, 
                     learning_rate: float, 
                     batch: int = 1):
            
            self.learning_rate = learning_rate
            self.batch = batch # Not used yet
        
        def net_update(self, 
                       inputs: List[float], 
                       net: List[List[Neuron]], 
                       output_error: List[float]):
            """Calls layer update function
            
            Calls layer update function on feature inputs and network

            Args:
                inputs (List[float]): Feature inputs to fit to
                net (List[List[Neuron]]): Network to update weights
                output_error (List[float]): Output error calculated during fitting
            """            
            
            self.layer_update(inputs, net, output_error)
        
        def layer_update(self, 
                         prev_layer_output: List[float], 
                         net: List[List[Neuron]], 
                         output_error: List[float]):
            """ Recursevely updates each layer
            
            Recursively updates each layer through gradient descent backpropagation

            Args:
                prev_layer_output (List[float]): Output from previous layer's neurons
                net (List[List[Neuron]]): Recursive level sub-network
                output_error (List[float]): Output error calculated during fitting

            Returns:
                List[float]: List of errors calculated on current layer for previous layer update 
            """            
            
            layer_error_list = []
            layer_output = [neuron.get_output(prev_layer_output) for neuron in net[0]]
            
            # Base run
            if len(net) == 1:
                for i, neuron in enumerate(net[0]):
                    gradient = neuron.activation_func.gradient

                    weights = neuron.get_weights()

                    error = output_error[i]*gradient(layer_output[i])
                    new_weights = [(weights[j] + self.learning_rate*error*input) for j, input in enumerate(prev_layer_output)]
                    bias_weight = weights[-1] + self.learning_rate*error*neuron.bias
                    new_weights.append(bias_weight)
                    neuron.set_weights(new_weights)
                    layer_error_list.append(error)

                return layer_error_list
            
            # Recursive run
            else:
                next_layer_error = self.layer_update(layer_output, net[1:], output_error)
                for i, neuron in enumerate(net[0]):
                    gradient = neuron.activation_func.gradient
                    weights = neuron.get_weights()
                    error = np.sum([next_layer_error[j]*next_layer_neuron.get_weights()[i] for j, next_layer_neuron in enumerate(net[1])])*gradient(layer_output[i])
                    new_weights = [(weights[k] + self.learning_rate*error*input) for k, input in enumerate(prev_layer_output)]
                    bias_weight = weights[-1] + self.learning_rate*error*neuron.bias
                    new_weights.append(bias_weight)
                    neuron.set_weights(new_weights)
                    layer_error_list.append(error)

                return layer_error_list

        
        
    class StochasticGradientDescent(GradientDescent):
        
        def __init__(self, 
                     learning_rate: float):
            
            self.learning_rate = learning_rate
        
    
class Models():
    
    class Sequential():
        
        def __init__(self, 
                     layers: List[Layers] = None):
            
            if layers is not None:
                self.topology = layers
            else:
                self.topology = []
                
            self.net = []
            self.layer_count = 0
            self._compiled = False
        
        def compile(self, 
                    optimizer: Optimizers,
                    loss: Losses,
                    seed: int = None):
            """ Compiles network topology

            Compiles network using described architecture and topology,
            instantiating neurons and operating functions 

            Args:
                optimizer (Optimizers): Optimizer to be used
                loss (Losses): Loss function to be used
                seed (int, optional): Seed to which generate random weights and shuffling training dataset. Defaults to None.

            Raises:
                ValueError: Input shape not defined
            """            
            
            if not seed:
                seed = np.random.randint(100000)
                
            self.seed = seed
            self.optimizer = optimizer
            self.loss = loss
            
            if not self.topology:
                raise
            
            if self.topology[0].input_shape is None:
                raise ValueError("No input shape defined. Add <Layers.Input> or set <input_shape> on first layer")
            
            self.input_shape = self.topology[0].input_shape
            self.output_shape = self.topology[-1].unit_count
            
            layer_input_shape = self.input_shape
            for layer in self.topology:
                if layer.type == "dense":
                    neuron_units = layer.initialize(layer_input_shape)
                    self.net.append(neuron_units)
                    self.layer_count = self.layer_count + 1
                    layer_input_shape = layer.output_shape
                    
                if layer.type == "normalization":
                    ...
            self.layer_count = len(self.net)
            
            self._compiled = True

            
        def fit(self, 
                dataset: pd.DataFrame,
                epochs: int,
                shuffle: bool = True,
                verbose: bool = True):
            """Fits network to training dataset

            Args:
                dataset (pd.DataFrame): Training dataset
                epochs (int): Epochs or iterations
                shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
                verbose (bool, optional): Whether to show epoch progress. Defaults to True.
                
            Raises:
                RuntimeError: Model not compiled
            """                      

            if not self._compiled:
                raise RuntimeError("Model not compiled")
            
            if shuffle:
                dataset = dataset.sample(frac = 1, random_state = self.seed)
            
            self.clear_weights()
            
            for i in range(epochs):
                print(f"Epoch {i+1}/{epochs}")
                
                start_time = time.perf_counter()
                for i in range(len(dataset)):
                    row = list(dataset.iloc[i,:])
                    
                    predicted = self.run_net(row[:-self.output_shape])
                    expected = row[-self.output_shape:]
                    
                    error = Error.Difference(expected, predicted)
                    
                    self.optimizer.net_update(row[:-self.output_shape], self.net, error)
                    
                    loss_val = self.loss.get_loss(dataset, self.net, self.output_shape,self.run_net)
                    
                    if verbose:
                        current_time = time.perf_counter()
                        progress = f"{i + 1}/{len(dataset)} {_equalsign_percent(bins = 30, value = i + 1, total = len(dataset))} - elapsed: {_perftime_str(current_time - start_time)} - loss {loss_val:.4f}"
                        print(progress, end="\r")

                print(end="\n")
                
        def predict(self, 
                    dataset: pd.DataFrame):
            """ Predicts network output
            
            Predicts network output over feature inputs dataset. 
            Dataset may be labeled  

            Args:
                dataset (pd.DataFrame): feature inputs to predict over

            Returns:
                pd.Dataframe: New dataframe with prediction columns added
            """            
            
            output = pd.DataFrame(columns = dataset.columns)
            
            labeled = False
            if len(dataset.columns) > self.input_shape:
                labeled = True
                
            for i in range(self.output_shape):
                # Take class names in case of predicting over labeled data
                if labeled:
                    output[f"predicted {dataset.columns[i- (self.output_shape)]}"] = pd.Series(dtype="float64")
                else: 
                    output[f"predicted {i}"] = pd.Series(dtype="float64")
                    
            for i in range(len(dataset)):
                row = list(dataset.iloc[i,:])
                predicted = self.run_net(row[:self.input_shape])
                row = row + predicted
                output.loc[len(output)] = row
                
            return output
        
        def add(self, 
                layer: Layers):
            """ Adds layer to network topology

            Args:
                layer (Layers): Layer object
            """            
            
            self.topology.append(layer)
            
        def run_net(self, 
                    inputs: List[float]):
            """ Runs net over input

            Args:
                inputs (List[float]): feature inputs to run net on 

            Returns:
                List[float]: List of end layer outputs 
            """            
            
            layer_vals = []
            layer_vals = list(inputs)
            for layer in self.net:
                layer_output = [neuron.get_output(layer_vals) for neuron in layer]
                layer_vals = layer_output
                
            return layer_vals
        
        def clear_weights(self):
            """ Reset weights
            
                Resets weights using network attributed seed
            """            
            
            np.random.seed(self.seed)
            for layer in self.net:
                for neuron in layer:
                    neuron.random_weights()
        
        def set_seed(self, 
                     seed: int):
            
            self.seed = seed
                    
        def get_seed(self):
            
            return self.seed
        
        def reset_seed(self):
            self.seed = np.random.seed()
            
        def net_weights(self):
            """ Displays network weights       
                 
            """  
                      
            for i, layer in enumerate(self.net):
                layer_weights = [f"neuron {i}: {neuron.get_weights()}" for i, neuron in enumerate(layer)]
                print(f"Layer {i} weights: {layer_weights}")
