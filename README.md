## XOR Neural Network in TypeScript

This repository contains a simple implementation of a Multi-Layer Perceptron (MLP) neural network in TypeScript. The MLP is designed to solve the XOR problem, a classic test for evaluating the capabilities of neural networks.

### Features

- Implementation of a neural network from scratch in TypeScript
- Utilizes He initialization for weight initialization
- Includes forward propagation and backpropagation algorithms
- Employs the Adam optimizer for training the neural network
- Designed to solve the XOR problem

### Getting Started

#### Prerequisites

- Node.js (version 18 or higher)

#### Step

1. Clone the repository:

   ```bash
   git clone https://github.com/ashc0/xor-neural-network-ts.git
   cd xor-neural-network-ts
   ```

2. Install the dependencies:

   ```bash
   npm install
   ```

3. Run the project:

   ```bash
   npm run start
   ```

### Project Structure

- `dataset/Xor_Dataset.csv`: XOR dataset downloaded from kaggle.
- `src/NeuralNetwork.ts`: Contains the implementation of the MLP neural network.
- `src/utils.ts`: Contains utility functions for matrix operations and activation functions.
- `src/index.ts`: The main entry point of the application, including training and evaluation logic.

### Example Output

```bash
Epoch 0: Loss = 0.2500, Accuracy = 50.00%
Epoch 1000: Loss = 0.0047, Accuracy = 100.00%
Epoch 2000: Loss = 0.0001, Accuracy = 100.00%
...
Input: 0,0, Predicted: 0.021, Target: 0
Input: 0,1, Predicted: 0.998, Target: 1
Input: 1,0, Predicted: 0.998, Target: 1
Input: 1,1, Predicted: 0.001, Target: 0
```
