from main_class import Value
import random
class Neuron:
  def __init__(self,nin):#nin is the number of input values for each nueron
    self.w=[Value(random.uniform(-1,1)) for _ in range(nin)]
    self.b=Value(random.uniform(-1,1))

  def __call__(self,x):#x is for w*x+b
    act=sum((wi * xi for wi,xi in zip(self.w,x)), self.b)#here b is the starting point for wx+b
    out=act.tanh()#act meaning the variable to be passed through the activation function, in this case it's tan
    return out

  def parameters(self):
    return self.w+[self.b]

class Layer:
  def __init__(self,nin,nout):#nout is the number of output nuerons
    self.nuerons=[Neuron(nin) for _ in range(nout)]

  def __call__(self,x):
    outs=[n(x) for n in self.nuerons]
    return outs[0] if len(outs)==1 else outs

  def parameter(self):
    return [p for nueron in self.nuerons for p in nueron.parameters()]

class MLP:
  def __init__(self,nin,nouts):
    sz=[nin]+nouts
    self.layers=[Layer(sz[i],sz[i+1]) for i in range(len(nouts))]

  def __call__(self,x):
    for layer in self.layers:
      x=layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameter()]


