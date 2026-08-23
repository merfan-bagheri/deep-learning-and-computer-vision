# 🎨 HW5: Deep Generative Modeling (VAE, CVAE & VQ-VAE)

**Author**: Muhammaderfan Bagherinejad

Exploration of latent variable generative models in deep learning, focusing on continuous variational manifolds (VAE, CVAE) and discrete codebook representations (Vector-Quantized VAE).

---

## 📌 Module Breakdown

### 🌀 Q1: Variational Autoencoders (VAE) & Conditional VAE (CVAE)
- **Notebook**: `q1/HW5_Q1_VAE_CVAE.ipynb`
- **Topics**:
  - Probabilistic encoder-decoder frameworks.
  - Reparameterization trick for end-to-end backpropagation through Gaussian latent spaces \( z \sim \mathcal{N}(\mu, \Sigma) \).
  - Conditional generation conditioning on class labels.
  - Latent space manifold traversal and smooth class-to-class interpolation.
- **Visual Demo**:
  - `q1/gif.gif`: Animated 2D latent space interpolation demonstrating continuous reconstruction transitions across digits/classes.

### 🔲 Q2: Vector-Quantized Variational Autoencoder (VQ-VAE)
- **Notebook**: `q2/HW5_Q2_VQ_VAE.ipynb`
- **Topics**:
  - Discrete latent representation learning avoiding "posterior collapse".
  - Codebook vector quantization with nearest-neighbor lookup in embedding space.
  - Straight-through gradient estimator (STE) and commitment loss formulation.
