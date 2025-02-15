import unittest
import random
from neural_net import Neuron,Layer,MLP

class TestNeuralNet(unittest.TestCase):
    def test_neuron_forward(self):
        n = Neuron(3)  # Neuron with 3 inputs
        x = [1.0, -2.0, 3.0]  
        output = n(x)  # Forward pass
        self.assertIsInstance(float(output.data), float)  # Output should be a single float

    def test_layer_forward(self):
        layer = Layer(3, 2)  # 3 input neurons, 2 output neurons
        x = [0.5, -1.5, 2.0] 
        output = layer(x)  # Forward pass
        self.assertEqual(len(output), 2)  

    def test_mlp_forward(self):
        mlp = MLP(3, [4, 2])  # MLP with 3 inputs, 2 layers (4 neurons, 2 neurons)
        x = [0.1, -0.2, 0.3]  
        output = mlp(x)  # Forward pass
        self.assertEqual(len(output), 2) 

if __name__ == "__main__":
    unittest.main()
