# Stable Diffusion

<img src='https://miro.medium.com/v2/resize:fit:2000/format:webp/1*pFNOzxb0_7WkcAyK5NhMxA.png' width='600'>

Diffusion models:
- use fuzzy noise to encode an image.
- use a noise predictor along with a reverse diffusion process to put the image back together.

Stable Diffusion (a Diffusion models):
- don't use the pixel space of the image.
- uses a simplified latent space.
- uses variational autoencoder (VAE) files in the decoder to capture intricate details such as eyes.

`Note:` 
- The effectiveness of the smaller latent space is based on the idea that natural images follow patterns rather than randomness.
-  **Variational Autoencoders (VAEs)**: VAEs are a type of generative model that learns to encode data into a compressed latent space and then decode it back into data. They are particularly useful for generating variations of the input data.

This project will cover:
1. **VAE**:
2. **Forward Diffusion**:
3. **Reverse Diffusion**:
4. **Noise Predictor (U-Net)**:
5. **Text Conditioning**:
