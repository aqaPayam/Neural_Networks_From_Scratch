
# Neural Networks From Scratch

This repository contains an implementation of neural networks from scratch using Numpy. The code is designed to help you understand the fundamentals of neural networks, including forward and backward propagation, gradient checking, and optimization algorithms. By building these components from scratch, you gain a deeper understanding of how neural networks operate under the hood.

## Repository Structure

- `Neural_Networks_From_Scratch_With_Numpy.ipynb`: Jupyter notebook that walks through building and training neural networks from scratch.
- `data.py`: Contains functions for loading and preprocessing data.
- `gradient_check.py`: Implements gradient checking to verify the correctness of backpropagation.
- `optim.py`: Contains various optimization algorithms like SGD and SGD with Momentum.
- `solver.py`: Manages the training loop, applying optimization algorithms, and evaluating the model.

## File Descriptions

### `Neural_Networks_From_Scratch_With_Numpy.ipynb`

This Jupyter notebook provides a comprehensive guide to building and training a neural network from scratch. The notebook covers the following sections:

1. **Introduction**: Overview of the notebook's objectives and the libraries used.
2. **Importing Required Libraries**: Importing Numpy and Matplotlib.
3. **Helper Functions and Classes**: 
    - **Initialization Functions**: Functions to initialize weights and biases of the neural network from scratch.
    - **Activation Functions**: Implementations of Sigmoid and ReLU activation functions and their derivatives from scratch.
    - **Loss Functions**: Implementation of Mean Squared Error (MSE) loss function and its derivative from scratch.
4. **Neural Network Class**: 
    - **Initialization**: Setting up the architecture of the network, including the number of layers and neurons, from scratch.
    - **Forward Propagation**: Computing the output of the network step-by-step from scratch.
    - **Backward Propagation**: Updating the weights and biases based on the error calculated from scratch.
    - **Training Method**: Implementing the training loop from scratch, adjusting weights based on the computed loss.
5. **Training the Network**: 
    - **Data Preparation**: Loading and normalizing the dataset from scratch.
    - **Network Initialization**: Creating an instance of the neural network with the specified architecture from scratch.
    - **Training Loop**: Running multiple iterations of forward and backward propagation to minimize the loss from scratch.
6. **Results Visualization**: Code to visualize the training loss and RMS error for both training and validation datasets.

#### Focus on "Scratch"

In this notebook, the term "scratch" emphasizes the manual implementation of neural network components without relying on high-level libraries like TensorFlow or PyTorch. The key aspects built from scratch include:

- **Weight and Bias Initialization**: Directly initializing the network parameters using Numpy arrays.
- **Activation Functions**: Writing the mathematical functions for Sigmoid and ReLU and their derivatives.
- **Forward Propagation**: Manually computing the activations for each layer.
- **Backward Propagation**: Calculating gradients for each layer and updating the weights and biases.
- **Optimization Algorithms**: Implementing basic optimization techniques like SGD and SGD with Momentum.
- **Training Loop**: Creating the loop to iterate through epochs and update the network parameters.

### `data.py`

This file contains functions to load and preprocess the data, ensuring it is ready for training the neural network from scratch.

- **Load Data Function**: Loads the dataset, normalizes it, and splits it into training, validation, and test sets.
- **Preprocessing Functions**: Functions to normalize, reshape, or augment data as needed.

### `gradient_check.py`

This file implements gradient checking to verify the correctness of the backpropagation implementation from scratch.

- **`grad_check_sparse`**: Compares the numerical gradient with the analytic gradient to ensure correctness, printing the relative error for randomly chosen elements.

### `optim.py`

This file contains various optimization algorithms used to update the weights of the neural network, implemented from scratch.

- **SGD (Stochastic Gradient Descent)**: Basic gradient descent algorithm to update weights.
- **SGD with Momentum**: An enhanced version of SGD to accelerate convergence by adding a momentum term.

### `solver.py`

This file contains a class that orchestrates the training process, managing the training loop, optimization, and evaluation from scratch.

- **`Solver`**: Manages the training process, including iterating through epochs, updating model parameters using specified optimization rules, and computing and printing loss and accuracy.

## Getting Started

### Prerequisites

Make sure you have the following installed:
- Python 3.x
- Numpy
- Matplotlib

### Running the Code

1. Clone the repository:
   ```sh
   git clone https://github.com/aqaPayam/Neural_Networks_From_Scratch.git
   cd Neural_Networks_From_Scratch
   ```

2. Install the required packages:
   ```sh
   pip install numpy matplotlib
   ```

3. Open the Jupyter notebook:
   ```sh
   jupyter notebook Neural_Networks_From_Scratch_With_Numpy.ipynb
   ```

4. Run the cells in the notebook to build and train the neural network from scratch.

## Usage

- Use the `Neural_Networks_From_Scratch_With_Numpy.ipynb` notebook to understand the step-by-step implementation of neural networks.
- Use the utility scripts (`data.py`, `gradient_check.py`, `optim.py`, `solver.py`) to extend or modify the functionality.

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have suggestions for improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
