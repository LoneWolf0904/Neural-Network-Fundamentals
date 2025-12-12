# Neural Network Fundamentals (From Scratch)

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

---

## 📂 Project 2: Optimized Architecture (Keras/TensorFlow)
**File:** `02_Optimized_ANN_Architecture.ipynb`

After establishing the mathematical baseline in Project 1, I implemented a scalable architecture using **Keras** to handle multi-class classification (4 possible NPC actions).

### 🧠 The Architecture
* **Input Layer:** 2 Neurons (Range to Enemy, Enemy Strength)
* **Hidden Layers:** 2 Layers (10 Neurons each) using **ReLU** activation to solve the vanishing gradient problem.
* **Output Layer:** 4 Neurons (Attack, Flock, Evade, Surrender) using **Softmax** for probability distribution.

### ⚡ Technical Improvements
* **Optimizer (Adam):** Replaced fixed learning rate with **Adam** (Adaptive Moment Estimation) for faster convergence.
* **Validation Strategy:** Implemented `train_test_split` (80/20) to validate the model on unseen data and prevent overfitting.
* **Loss Function:** utilized **Categorical Crossentropy**, the standard for multi-class classification tasks.

### 📊 Results
* **Accuracy:** Achieved **100% (1.0)** on the validation set.
* **Loss:** Reduced to **~0.002**, demonstrating highly confident predictions compared to the manual implementation.

---

## 📂 Project 3: Generative AI & Transfer Learning
**File:** `03_Generative_AI_Stable_Diffusion.ipynb`

Moving beyond classification, this project explores **Generative AI** by deploying a pre-trained Latent Diffusion Model (LDM) for text-to-image synthesis.

### 🤖 The Tech Stack
* **Library:** Hugging Face `diffusers` & `transformers`.
* **Model:** Stable Diffusion (Pipeline).
* **Optimization:** PyTorch with CUDA acceleration and FP16 (half-precision) to reduce VRAM usage.

### ⚙️ How it Works
1.  **Text Encoding:** The prompt ("A Cat Face") is converted into vector embeddings.
2.  **Latent Diffusion:** The model starts with random noise and iteratively "denoises" it, guided by the text embeddings, to form a coherent image in latent space.
3.  **Decoding:** The VAE (Variational Autoencoder) decodes the latent representation into the final pixel-art image.

### 💻 Code Snippet (GPU Acceleration)
```python
# optimizing for GPU memory and speed using half-precision (float16)
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

image = pipe(prompt).images[0]
```
## 🎨 Outcome
Successfully generated high-quality images from natural language prompts, demonstrating proficiency with Modern AI Pipelines and Transfer Learning.

