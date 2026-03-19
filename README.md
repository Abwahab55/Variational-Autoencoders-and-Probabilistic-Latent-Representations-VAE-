# Variational Autoencoder (VAE) on MNIST

> Probabilistic generative model implementing VAE with PyTorch on the MNIST handwritten digit dataset.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## Overview

This project implements a **Variational Autoencoder (VAE)** using PyTorch, applied to the MNIST handwritten digit dataset. VAEs are generative models that learn probabilistic latent representations of data by optimizing the **Evidence Lower Bound (ELBO)**.

Unlike standard autoencoders that encode inputs to a single point, VAEs encode inputs into a **distribution over latent variables** — enabling better generalization and the ability to generate new, realistic samples.

---

## Key Concepts

| Concept | Description |
|---|---|
| **Encoder** | Maps input images to mean (μ) and log-variance (σ²) of the latent distribution |
| **Reparameterization trick** | Enables backpropagation through stochastic sampling |
| **Decoder** | Reconstructs images from sampled latent vectors |
| **ELBO loss** | Reconstruction loss (BCE) + KL divergence regularization |
| **Latent sampling** | Generate new digits by sampling from the learned latent space |

---

## Project Structure

```
Variational-Autoencoders-and-Probabilistic-Latent-Representations-VAE-/
│
├── VAE.py        # Full VAE implementation — model, training loop, sampling & visualization
└── VAE.txt       # Project description and notes
```

---

## Model Architecture

```
Input (28×28)
     │
     ▼
┌─────────────┐
│   Encoder   │  → Fully connected layers
│             │  → Outputs: μ (mean), log σ² (log variance)
└─────────────┘
     │
     ▼  Reparameterization: z = μ + σ · ε,  ε ~ N(0, I)
     │
┌─────────────┐
│   Decoder   │  → Fully connected layers
│             │  → Output: Reconstructed image (28×28)
└─────────────┘
```

### Loss Function

```
ELBO Loss = Reconstruction Loss + KL Divergence

Reconstruction:  BCE(x_reconstructed, x_input)
KL Divergence:   -0.5 × Σ(1 + log σ² - μ² - σ²)
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/Abwahab55/Variational-Autoencoders-and-Probabilistic-Latent-Representations-VAE-.git
cd Variational-Autoencoders-and-Probabilistic-Latent-Representations-VAE-
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install torch torchvision matplotlib numpy
```

---

## Usage

### Run the VAE

```bash
python VAE.py
```

This will automatically:
1. Download the MNIST dataset
2. Train the VAE model
3. Reconstruct sample digits
4. Generate new digit images by sampling from the latent space

---

## Training Details

| Parameter | Value |
|---|---|
| Dataset | MNIST (60,000 train / 10,000 test) |
| Input size | 28 × 28 (784 flattened) |
| Latent dimension | 20 |
| Batch size | 128 |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 50 |

---

## Results

After training, the model can:

- **Reconstruct** input digits with high fidelity
- **Generate** new digit-like images by sampling `z ~ N(0, I)`
- **Interpolate** smoothly between digits in latent space

---

## Background

Variational Autoencoders were introduced by Kingma & Welling (2013) in *"Auto-Encoding Variational Bayes"*. They combine:

- **Variational inference** from Bayesian statistics
- **Deep learning** for scalable inference networks

This makes them a foundational model for unsupervised representation learning and generative deep learning.

---

## Author

**Abdul Wahab**  
AE @ Lumissil Microsystems | SiC power systems → Cloud computing | 5 IEEE publications

- GitHub: [@Abwahab55](https://github.com/Abwahab55)
- Email: wahab.engr55@yahoo.com

---

## Acknowledgements

- [Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114) — Original VAE paper
- [PyTorch](https://pytorch.org/) — Deep learning framework
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — Yann LeCun et al.
