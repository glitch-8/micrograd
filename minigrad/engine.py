import math


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0 # gradients of Value object with respect to the final leaf node
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

        self.supported_activations = [
            'tanh',
            'sigmoid',
            'relu',
            'identity'
        ]

    
    def __repr__(self):
        return f'Value(data={self.data}, grad={self.grad}, label={self.label})'
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad

        out._backward = _backward

        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data*other.data, (self, other), '*')

        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += (other*self.data**(other - 1))*out.grad
        
        out._backward = _backward

        return out
    
    def __lt__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data < other.data

    def __gt__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data > other.data

    def __le__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data <= other.data
    
    def __ge__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data >= other.data
    
    def __eq__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self.data == other.data

    def __hash__(self):
        return id(self)
    
    def tanh(self):
        n = self.data
        # t = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)
        t = math.tanh(n)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2)*out.grad
        
        out._backward = _backward

        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward

        return out

    def log(self):
        x = self.data
        out = Value(math.log(x), (self, ), 'ln')

        def _backward():
            self.grad += (1/x)*out.grad
        
        out._backward = _backward

        return out
    
    def relu(self):
        x = self.data
        out = Value(max(x, 0), (self, ), 'ReLU')

        def _backward():
            self.grad += (1 if x>0 else 0)*out.grad
        
        out._backward = _backward

        return out
    
    def sigmoid(self):
        x = self.data
        t = 1/(1 + math.exp(-x))
        out = Value(t, (self, ), 'Sigmoid')

        def _backward():
            self.grad += t * (1 - t) * out.grad
        
        out._backward = _backward

        return out
    
    def identity(self):
        out = Value(self.data, (self, ), 'id')

        def _backward():
            self.grad += out.grad
        
        out._backward = _backward

        return out
    
    def min(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(min(self.data, other.data), (self, other), 'min')

        def _backward():
            self.grad += (1 if self.data<other.data else 0)*out.grad
            other.grad += (1 if other.data<self.data else 0)*out.grad
        
        out._backward = _backward

        return out
    
    def max(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(max(self.data, other.data), (self, other), 'max')

        def _backward():
            self.grad += (1 if self.data>other.data else 0)*out.grad
            other.grad += (1 if other.data>self.data else 0)*out.grad
        
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        
        for node in reversed(topo):
            node._backward()