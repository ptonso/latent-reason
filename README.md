# 🧠 Variational Autoencoder Playground

A modular and extensible framework for training, evaluating, and visualizing Variational Autoencoders (VAEs), including β-VAE variants. Built for research and experimentation across datasets such as CelebA, MNIST, dSprites, 3D Shapes, and more.

---

## 🔧 Features

- 💾 **Flexible Data Loading** via YAML config files (`train`, `val`, `test` paths).
- 🧱 **Modular Architecture** with customizable encoder/decoder configs.
- 🔬 **Evaluation Suite**:
  - Reconstruction visualization
  - Latent space PCA and dimension traversal
  - Latent statistics (μ, σ², KL contribution)
- 📈 **Training Engine** with:
  - Learning rate schedulers (`plateau`, `cosine`)
  - Early stopping and warm-up β
  - Resume from checkpoints
  - Loss and LR plots
- 🧪 **Dataset Preparation Scripts** for CelebA, CIFAR-100, dSprites, COIL-100, ModelNet40, and 3D Shapes.


