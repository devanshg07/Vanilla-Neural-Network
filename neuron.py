import random
import math
from typing import Union

class Scalar:
    """A numerical scalar with backward pass and gradient computation."""
    def __init__(self, val, _children=(), _op="", tag=""):
        self.val = val
        self._backward_fn = lambda: None
        self.tag = tag
        self.grad = 0
        self._ancestors = _children
        self._operation = _op

    def __repr__(self):
        return f"Scalar(val={self.val})"

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.val + other.val, (self, other), "+")
        def _backward_fn():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward_fn = _backward_fn
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.val * other.val, (self, other), "*")
        def _backward_fn():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad
        out._backward_fn = _backward_fn
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float"
        out = Scalar(self.val ** other, (self,), f"^{other}")
        def _backward_fn():
            self.grad += other * self.val ** (other-1) * out.grad
        out._backward_fn = _backward_fn
        return out

    def __truediv__(self, other):
        return self * (other**-1)

    def __lt__(self, other):
        return self.val < other.val

    def __gt__(self, other):
        return self.val > other.val

    def __le__(self, other):
        return self.val <= other.val

    def __ge__(self, other):
        return self.val >= other.val

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self):
        return id(self)

    def __ne__(self, other):
        return self.val != other.val

    def exp(self):
        x = self.val
        out = Scalar(math.exp(x), (self, ), "exp")
        def _backward_fn():
            self.grad += out.val * out.grad
        out._backward_fn = _backward_fn
        return out

    def tanh(self):
        x = self.val
        try:
            t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        except OverflowError:
            t = 1.0 if x > 0 else -1.0
        out = Scalar(t, (self, ), "tanh")
        def _backward_fn():
            self.grad += (1 - t**2) * out.grad
        out._backward_fn = _backward_fn
        return out

    def relu(self):
        x = self.val
        out = Scalar(max(0, x), (self, ), "relu")
        def _backward_fn():
            self.grad = out.grad * (x > 0)
        out._backward_fn = _backward_fn
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._ancestors:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward_fn()

class Node:
    def __init__(self, num_inputs: int, activation: str):
        self.weights = [Scalar(random.uniform(-1, 1), tag="w") for _ in range(num_inputs)]
        self.bias = Scalar(random.uniform(-1, 1), tag="b")
        self.activation = activation
    def __call__(self, x: list[Union[float, Scalar]]):
        act = sum([w*xv for w, xv in zip(self.weights, x)], self.bias)
        if self.activation == "relu":
            return act.relu()
        elif self.activation == "tanh":
            return act.tanh()
        return act
    def get_params(self):
        return self.weights + [self.bias]

class Block:
    def __init__(self, num_inputs: int, num_outputs: int, activation: str):
        self.nodes = [Node(num_inputs, activation) for _ in range(num_outputs)]
    def __call__(self, x: list[Union[float, Scalar]]):
        return [n(x) for n in self.nodes]
    def get_params(self):
        params = []
        for n in self.nodes:
            params.extend(n.get_params())
        return params

class NeuralNet:
    layer_type_choices = ["relu", "linear", "tanh"]
    def __init__(self, *layers: (str, int)):
        sz = layers
        self.blocks = []
        for i in range(len(sz)-1):
            nin, nout = sz[i][0], sz[i+1][0]
            act = sz[i+1][1]
            self.blocks.append(Block(nin, nout, act))
    def __call__(self, x: list[Union[float, Scalar]]):
        for block in self.blocks:
            x = block(x)
        return x[0] if len(x) == 1 else x
    def get_params(self):
        params = []
        for block in self.blocks:
            params.extend(block.get_params())
        return params
    def zero_grad(self):
        for p in self.get_params():
            p.grad = 0
    def nudge(self, step: float):
        for p in self.get_params():
            p.val += -step * p.grad

def _optim_sum(nums: list[Union[float, Scalar]]):
    total = nums[0]
    for n in nums[1:]:
        total = total + n
    return total

def mean_squared_error(ys: list[Union[float, Scalar]], ypred: list[Union[float, Scalar]]):
    n = len(ys)
    return _optim_sum([(y - yhat) ** 2 for y, yhat in zip(ys, ypred)]) * (1.0 / n)

def mean_abs_error(ys: list[Union[float, Scalar]], ypred: list[Union[float, Scalar]]):
    n = len(ys)
    return _optim_sum([abs(y - yhat) for y, yhat in zip(ys, ypred)]) * (1.0 / n)

def one_hot(label, class_list: list):
    return [1 if label == c else 0 for c in class_list]

