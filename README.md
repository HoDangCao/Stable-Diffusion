# Stable Diffusion

Welcome to the **Stable Diffusion** project! This repository contains a comprehensive implementation of Stable Diffusion, a generative model that leverages diffusion processes to create images. Stable Diffusion uses a combination of noise prediction, reverse diffusion, and text conditioning to produce unique and visually appealing outputs.

---

## Dataset

This project will be trained using example [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

Therefore, the pretrained model just can create digit images. For instance:

Run `python generate.py --prompt 8`

Output:

<img src='https://github.com/user-attachments/assets/64d6c40c-96fa-4306-ae8b-c8a9cb7fc6c3' width=300>


**Note**:
- For more diverse generated images, please re-train model with a richer dataset.
- For more detailed and correct images, please re-train model with a higher number of epochs (> 20 epochs).

---
## 📋 Key Features

1. **Diffusion Models**:
   - Use noise to encode an image and reverse the process to reconstruct it.
   - Implement forward and reverse diffusion mechanisms for training and sampling.

2. **Stable Diffusion Specifics**:
   - Operates in a simplified latent space instead of pixel space.
   - Utilizes **Variational Autoencoders (VAEs)** to preserve intricate image details.
   - Supports **text conditioning** via CLIP tokenization for precise image generation.

3. **Custom Architectures**:
   - **U-Net Architecture** with **skip connections** for noise prediction.
   - Variants: Concatenation-based and Addition-based skip connections.

4. **Attention Mechanisms**:
   - Incorporates **Cross Attention** and **Spatial Transformers** for contextual and spatial learning.

5. **Sampler Options**:
   - Implements the **Euler–Maruyama method** for efficient image sampling.

---

## 🏗️ Architecture Overview

### **Main Components**
1. **Variational Autoencoder (VAE)**:
   - Compresses 512×512 images to 64×64 in latent space (encoder) and restores them to full size (decoder).

2. **Forward Diffusion**:
   - Adds Gaussian noise to an image iteratively, used primarily during training.

3. **Reverse Diffusion**:
   - Reconstructs images by removing noise progressively.

4. **Noise Predictor (U-Net)**:
   - Uses a U-Net-based Residual Neural Network (ResNet) for denoising.

5. **Text Conditioning**:
   - Processes prompts with a CLIP tokenizer and embeds them into vectors for U-Net integration.

---

## 🧮 Key Mathematical Concepts

### **Forward Diffusion**
$$x(t+\Delta t) = x(t) + \sigma(t)\sqrt{\Delta t}\cdot r$$  
- Adds noise incrementally to a sample.

### **Reverse Diffusion**
$$x(t+\Delta t) = x(t) + \sigma^2(T-t)\cdot s(x,T-t)\Delta t + \sigma(T-t)\sqrt{\Delta t}\cdot r$$  
- Reverses the noise addition using the score function.

### **Score Function Learning**
- Neural networks are trained to predict the noise added during diffusion using the objective:
$$J = \mathbb{E} \left[\|s(x_{\text{noised}}, t) \sigma^2(t) + (x_{\text{noised}} - x_0)\|^2_2 \right]$$

---

## 🖥️ Implementation Highlights

### **U-Net Architecture**
- Encoder-decoder structure with skip connections for efficient image reconstruction.
- Two variants:
  - **Concatenation-based skip connections**.
  - **Addition-based skip connections** (offers better performance).

### **Attention Layers**
1. **Cross Attention**: Handles context-based sequence attention.
2. **Transformer Block**: Combines attention with neural networks.
3. **Spatial Transformer**: Converts spatial tensors to sequential form for transformers.

---

## 🚀 Training and Sampling

### **Training Steps**
1. Add noise to input images based on the forward diffusion process.
2. Train the model to predict noise levels at each time step using the loss function.

### **Sampling Steps**
1. Start with random noise.
2. Iteratively denoise the image using the trained noise predictor and reverse diffusion.

---

## 🔬 Visualization

### **Forward Diffusion**
- Visualizes the progressive addition of noise to an image.
  
### **Reverse Diffusion**
- Demonstrates how the model reconstructs images step-by-step.

### **Attention Mechanisms**
- Explains the role of cross and self-attention in image-text conditioning.

---

## 🔧 Setup and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/HoDangCao/Stable-Diffusion-Text2Img.git
   cd Stable-Diffusion-Text2Img
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate samples:
   ```bash
   python generate.py --prompt 1
   ```

4. For more manipulation such as training, adjust model, etc; please visit `Stable_Diffusion.ipynb` file.

---

## 📂 Directory Structure

```
stable-diffusion/
├── data/                   # Dataset and preprocessed files
├── models/                 # Model architectures
├── Stable_Diffusion.ipynb  # Full project manipulation
├── requirements.txt        # Necessary libraries
└── README.md               # Project documentation
```

## 🤝 Contributing
Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve this repository.

---

Happy generating! 🎨✨
