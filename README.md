# Micro_grad

Micro_grad is a lightweight implementation of a neural network from scratch, inspired by the "makemore" series by Andrej Karpathy. This project builds a simple multilayer perceptron (MLP) using a custom autograd system similar to TinyGrad.

## Features

* Implements fundamental components of a neural network:
  * **Neuron** : Represents a single computational unit with weights and biases.
  * **Layer** : A collection of neurons forming a neural network layer.
  * **MLP (Multi-Layer Perceptron)** : A stack of layers forming a full neural network.
* Uses a custom **Value class** to enable autodifferentiation for backpropagation.
* Written in **pure Python** without external deep learning frameworks.

## Installation

Clone this repository:

```bash
git clone https://github.com/Twahaaa/Micro_grad.git
cd Micro_grad
```

Ensure you have Python 3 installed, and create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt  # If you have dependencies
```

## Usage

Run the neural network implementation:

```bash
python main_class.py
```

Run tests:

```bash
python -m unittest test_neural_net.py
```

## File Structure

```
Micro_grad/
│── main_class.py         # Main entry point for the project
│── neural_net.py         # Neural network implementation
│── test_neural_net.py    # Unit tests for the neural network
│── README.md             # Project documentation
│── tinygrad_venv/        # Virtual environment (optional)
```

## References

This project is inspired by:

* [&#34;The spelled-out intro to neural networks and backpropagation&#34;](https://www.youtube.com/watch?v=VMj-3S1tku0) by Andrej Karpathy

## License

This project is open-source and available under the MIT License.
