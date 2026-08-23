# 🎨 Deep Generative Modeling (VAE, CVAE & VQ-VAE)

**Author**: Muhammaderfan Bagherinejad

Exploration of latent variable generative models in deep learning, focusing on continuous variational manifolds (VAE, CVAE) and discrete codebook representations (Vector-Quantized VAE).

---

## 📌 Module Breakdown

### 🌀 01. Variational Autoencoders (VAE) & Conditional VAE (CVAE)
- **Directory**: `01-VAE-and-CVAE-Latent-Interpolation/`
- **Notebook**: `01-VAE-and-CVAE-Latent-Interpolation/HW5_Q1_VAE_CVAE.ipynb`
- **Topics**:
  - Probabilistic encoder-decoder frameworks.
  - Reparameterization trick for end-to-end backpropagation through Gaussian latent spaces $z \sim \mathcal{N}(\mu, \Sigma)$.
  - Conditional generation conditioning on class labels.
  - Latent space manifold traversal and smooth class-to-class interpolation.
- **Visual Demo**:
  - `01-VAE-and-CVAE-Latent-Interpolation/gif.gif`: Animated 2D latent space interpolation demonstrating continuous reconstruction transitions across digits/classes.

### 🔲 02. Vector-Quantized Variational Autoencoder (VQ-VAE)
- **Directory**: `02-VQ-VAE-Discrete-Representations/`
- **Notebook**: `02-VQ-VAE-Discrete-Representations/HW5_Q2_VQ_VAE.ipynb`
- **Topics**:
  - Discrete latent representation learning avoiding "posterior collapse".
  - Codebook vector quantization with nearest-neighbor lookup in embedding space.
  - Straight-through gradient estimator (STE) and commitment loss formulation.
