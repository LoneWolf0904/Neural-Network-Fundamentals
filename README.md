Neural Network Fundamentals (From Scratch)

This repository explores the mathematical foundations of Deep Learning by implementing Artificial Neural Networks (ANNs) without high-level abstractions like TensorFlow or PyTorch.

## 📂 Project 1: Manual ANN Implementation
**File:** `01_Manual_ANN_Implementation.ipynb`

A custom 2-layer Neural Network built using **NumPy** to solve a non-linear classification problem (XOR Logic) for Game AI decision-making (Attack vs. Flee).

### 🧠 The Architecture
* **Input Layer:** 2 Neurons (NPC Power, Enemy Power)
* **Hidden Layer:** 2 Neurons
* **Output Layer:** 1 Neuron (Decision: Attack vs. Flee)
* **Activation Function:** Sigmoid

### 🧪 Technical Implementation Details
Based on my experimentation documented during development:

* **Activation Function (Sigmoid):** I chose Sigmoid over Step or Sign functions because it is differentiable, which is required for calculating gradients during Backpropagation. It effectively "squashes" the output between 0 and 1.
* **Epochs:** The model is trained over **10,000 epochs**. Testing showed this was the optimal number for the error to converge to zero.
* **Learning Rate (Alpha):**
    * I experimented with `1.0`, `0.1`, and `0.01`.
    * At `1.0`, the training was unstable and failed to settle.
    * At `0.01`, convergence was too slow.
    * **Result:** `0.1` was selected as the optimal hyperparameter.

### 🧮 The Mathematics (Backpropagation)
Instead of using `model.fit()`, I manually implemented the weight updates using Matrix Calculus:

1.  **Feed Forward:**
    $$Z = X \cdot W + B$$
    $$A = \sigma(Z)$$
2.  **Error Calculation:**
    $$E = Y_{target} - Y_{predicted}$$
3.  **Backpropagation (Chain Rule):**
    $$\frac{\partial E}{\partial W} = \delta \cdot A^{T}$$

### 💻 Code Snippet
```python
# Updating weights based on the calculated error gradients (Chain Rule)
w_hidden_output += hidden_output.T.dot(op_error) * alpha
w_input_hidden += X.T.dot(hidden_error) * alpha
``` 

### 📊 Results
The network successfully learned the XOR pattern:

* Equal Power (1,1 or 0,0): NPC chooses to Attack.

* Disparate Power (1,0 or 0,1): NPC chooses to Flee.

