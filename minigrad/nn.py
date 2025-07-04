import random
import math

from micrograd.engine import Value


class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0
    
    def parameters(self):
        return list()
    

class Neuron(Module):

    def __init__(self, nin, activation='tanh'):
        # based on activation function change
        # the bounds of random initialization to handle
        # vanishing gradients problem
        self.activation = activation
        if activation == 'relu':
            bound = math.sqrt(2/nin)
        else:
            bound = math.sqrt(1/nin)

        self.w = [Value(random.uniform(-bound, bound), label = f'w{counter}') for counter in range(nin)]

        # have a slightly positive bias for relu, for preventing dead neurons
        self.b = Value(0.1 if activation == 'relu' else 0, label='bias')

    def __call__(self, x):
        x = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

        # dynamically select which activation function to call based on input passed
        if self.activation not in x.supported_activations:
            raise ValueError(
                f'Invalid value passed for activation function.',
                f'Accepted values: {", ".join(x.supported_activations)}'
            )
        
        out = getattr(x, self.activation)()

        return out
    
    def parameters(self):
        return self.w + [self.b]
    

class Layer(Module):

    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params


class MLP(Module):

    def __init__(self, nin, nouts, activation='tanh', final_layer_activation='identity'):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], activation=activation) for i in range(len(nouts) - 1)
        ]
        # don't apply activation on last layer
        self.layers.append(Layer(sz[-2], sz[-1], activation=final_layer_activation))
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def parameters(self):
        params = list()
        for layer in self.layers:
            params.extend(layer.parameters())
        
        return params