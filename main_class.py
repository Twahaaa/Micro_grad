import math
class Value:
  def __init__(self,data, _children=(),_op='',label=''):
    self.data=data
    self.grad=0.0
    self._backward=lambda:None
    self.prev=set(_children)
    self._op=_op
    self.label=label

  def __repr__(self):
    return f"Value(data={self.data})"

  def __add__(self,other):
    other=other if isinstance(other, Value) else Value(other)
    out = Value(self.data+other.data, (self, other),'+')
    def _backward():
      self.grad+=1.0*out.grad # '+=' because in case of multivariate case (where 2 nodes use the same value like b=a+c d=b+a, we accumulate the gradients or else it'll be initialised to 1)
      other.grad+=1.0*out.grad
    out._backward=_backward
    return out


  def __mul__(self,other):
    other=other if isinstance(other, Value) else Value(other)
    out = Value(self.data*other.data, (self, other), '*')
    def _backward():
      self.grad+=other.data*out.grad
      other.grad+=self.data*out.grad
    out._backward=_backward
    return out

  def __pow__(self,other):
    assert isinstance(other, (int,float)) #The reason why its 'other' and not 'other.data' is becase the power should always be an int or a float not an object ad that's why the assertiion is there
    out=Value(self.data**other, (self, ), f'**{other}')
    def _backward():
      self.grad+= other*(self.data**other-1)*out.grad
    out._backward=_backward
    return out


  def __radd__(self,other):
    return self+other

  def __rmul__(self,other):
    return self*other

  def __truediv__(self,other):
    return self*other**-1

  def __neg__(self):
    return self*-1

  def __sub__(self,other):
    return self+(-other)

  def __rsub__(self,other):
    return self+(-other)

  def tanh(self):
    x=self.data
    t=(math.exp(2*x)-1)/(math.exp(2*x)+1)
    out=Value(t,(self,),'tanh')
    def _backward():
      self.grad+=(1-t**2)*out.grad
    out._backward=_backward
    return out

  def exp(self):
    x=self.data
    out=Value(math.exp(x),(self, ),'exp')
    def _backward():
      self.grad+=out.data*out.grad
    out._backward=_backward
    return out

  def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v.prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad=1.0
    for node in reversed(topo):
      node._backward()